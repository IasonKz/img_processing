"""
Generates (STEP, units mm):
  drain_comsol.step        drain segment (solid) + 9 channel lumens (fluid domains), conformal
  tissue_realistic.step    organic soft-tissue block (fat + muscle, undulating interface)
                           with a tunnel that matches the drain envelope exactly
  assembly_drain_tissue.step   all of the above, conformal (shared faces merged)

Drain: simplified, COMSOL-friendly analogue of the USZ design (not the real section):
elliptical body 18 x 10 mm, 8 radial slots + central channel, 30 mm long segment,
smallest feature 0.9 mm, no fillets / chamfers / slivers.
Tissue: lump whose outline varies smoothly along z (low-frequency Fourier noise),
fat above, muscle below, interface is a freeform undulating surface.
"""
import sys
import numpy as np
import gmsh

P = dict(L=30.0,
         a=9.0, b=5.0, n_slots=8, slot_rx=1.4, slot_ry=0.45, ring=0.58, core_r=0.7,
         A=30.0, B=20.0, n_sec=7, y_if=-9.5, seed=7)

rng = np.random.default_rng(P["seed"])
MODES = [(2, 0.060), (3, 0.040), (4, 0.025), (5, 0.015)]
PHI = {m: rng.uniform(0, 2 * np.pi) for m, _ in MODES}
KAP = {m: rng.uniform(-1.0, 1.0) for m, _ in MODES}
CTR = rng.uniform(0, 2 * np.pi, 2)
FIF = rng.uniform(0, 2 * np.pi, 3)


def outline(theta, z):
    t = z / P["L"]
    r = 1.0 + sum(amp * np.cos(m * theta + PHI[m] + np.pi * KAP[m] * t) for m, amp in MODES)
    r = r * (1.0 + 0.05 * np.sin(np.pi * t))
    cx = 1.5 * np.sin(2 * np.pi * t + CTR[0])
    cy = 1.0 * np.sin(2 * np.pi * t + CTR[1])
    return cx + P["A"] * r * np.cos(theta), cy + P["B"] * r * np.sin(theta)


def interface(x, z):
    t = z / P["L"]
    return (P["y_if"] + 1.3 * np.sin(2 * np.pi * x / 24 + FIF[0] + 1.5 * t)
            + 0.8 * np.cos(2 * np.pi * x / 13 + FIF[1] - 1.0 * t)
            + 0.5 * np.sin(2 * np.pi * z / 17 + FIF[2]))


occ = None


def vol_of(out):
    return [t for d, t in out if d == 3][0]


def drain_parts():
    body = occ.addDisk(0, 0, 0, P["a"], P["b"])
    chans = []
    for k in range(P["n_slots"]):
        th = 2 * np.pi * k / P["n_slots"]
        x, y = P["ring"] * P["a"] * np.cos(th), P["ring"] * P["b"] * np.sin(th)
        d = occ.addDisk(x, y, 0, P["slot_rx"], P["slot_ry"])
        occ.rotate([(2, d)], x, y, 0, 0, 0, 1, np.arctan2(y, x))
        chans.append(d)
    chans.append(occ.addDisk(0, 0, 0, P["core_r"], P["core_r"]))
    bv = vol_of(occ.extrude([(2, body)], 0, 0, P["L"]))
    cv = [vol_of(occ.extrude([(2, c)], 0, 0, P["L"])) for c in chans]
    return bv, cv


def lump_volume():
    wires = []
    n = 18
    for z in np.linspace(0, P["L"], P["n_sec"]):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = [occ.addPoint(*outline(t, z), z) for t in th]
        pts.append(pts[0])
        s = occ.addSpline(pts)
        wires.append(occ.addWire([s]))
    return vol_of(occ.addThruSections(wires, makeSolid=True, makeRuled=False))


def upper_block():
    wires = []
    zs = np.linspace(-2.0, P["L"] + 2.0, P["n_sec"] + 2)
    for z in zs:
        xs = np.linspace(-45, 45, 13)
        pts = [occ.addPoint(x, interface(x, z), z) for x in xs]
        s = occ.addSpline(pts)
        p_ur = occ.addPoint(45, 35, z)
        p_ul = occ.addPoint(-45, 35, z)
        l1 = occ.addLine(pts[-1], p_ur)
        l2 = occ.addLine(p_ur, p_ul)
        l3 = occ.addLine(p_ul, pts[0])
        wires.append(occ.addWire([s, l1, l2, l3]))
    return vol_of(occ.addThruSections(wires, makeSolid=True, makeRuled=False))


def tissue_layers():
    lump = lump_volume()
    upper = upper_block()
    lump2 = occ.copy([(3, lump)])
    fat, _ = occ.intersect([(3, lump)], [(3, upper)], removeObject=True, removeTool=False)
    musc, _ = occ.cut(lump2, [(3, upper)], removeObject=True, removeTool=True)
    return fat, musc


def classify():
    """Return dict name -> list of volume tags, from mass and centre of mass."""
    res = dict(fat=[], muscle=[], drain=[], channels=[])
    vols = [t for d, t in gmsh.model.getEntities(3)]
    for t in vols:
        m = occ.getMass(3, t)
        x, y, z = occ.getCenterOfMass(3, t)
        if m < 100:
            res["channels"].append(t)
        elif abs(x) < 1.5 and abs(y) < 1.5 and m < 5000:
            res["drain"].append(t)
        elif y < -8:
            res["muscle"].append(t)
        else:
            res["fat"].append(t)
    return res


def fresh(name):
    global occ
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ


def write(path):
    occ.synchronize()
    gmsh.write(path)


def build(kind, outdir):
    fresh(kind)
    if kind == "drain":
        bv, cv = drain_parts()
        occ.fragment([(3, bv)], [(3, c) for c in cv])
        write(f"{outdir}/drain_comsol.step")
    elif kind == "tissue":
        fat, musc = tissue_layers()
        bv, cv = drain_parts()
        occ.remove([(3, c) for c in cv], recursive=True)
        fat_cut, _ = occ.cut(fat, [(3, bv)], removeObject=True, removeTool=True)
        occ.fragment(fat_cut + musc, [])
        write(f"{outdir}/tissue_realistic.step")
    else:
        fat, musc = tissue_layers()
        bv, cv = drain_parts()
        occ.fragment(fat + musc + [(3, bv)], [(3, c) for c in cv])
        write(f"{outdir}/assembly_drain_tissue.step")
    return classify()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    for k in ("drain", "tissue", "assembly"):
        cl = build(k, out)
        print(k, {n: len(v) for n, v in cl.items()})
    gmsh.finalize()
