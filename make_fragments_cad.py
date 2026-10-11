"""
Perforated drain + surrounding fluid cavity + irregular tissue fragments (STEP, mm).

Outputs (in outdir):
  drain_perforated.step           drain solid: elliptical body 18x10 mm, 30 mm long, 8 radial slot channels
                                  (blind at z<1.5, open at z=30) + central lumen, 0.8 mm side holes
  fragments_only.step             all tissue fragments as separate solids
  drain_fluid_fragments.step      conformal assembly: fluid (1 volume, includes channels and holes),
                                  drain solid, fragments

Hole diameter is HOLE_D. Fragments are 1 to 10 hole diameters in their largest dimension:
  strands   long and slender, curved
  flakes    thin irregular plates
  blobs     compact irregular lumps
  branched  two strands fused in a V/Y shape
Placement is random (seeded), non-overlapping, kept clear of the drain and of the cavity wall;
a few are deliberately parked right next to holes.
"""
import sys
import numpy as np
import gmsh

SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = np.random.default_rng(SEED)

L = 30.0
A_D, B_D = 9.0, 5.0                       # drain ellipse semi-axes
N_SLOTS, SLOT_RX, SLOT_RY, RING, CORE_R = 8, 1.4, 0.45, 0.58, 0.7
HOLE_D = 0.8
HOLE_Z = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
CORE_HOLE_Z = [10.0, 18.0, 26.0]
CH_Z0 = 1.5                               # channels are blind below this z
CLEAR = 0.35                              # minimum gap between any two solids (mm)
CAV_A, CAV_B = 24.0, 17.0                 # cavity semi-axes before noise
MODES = [(2, 0.06), (3, 0.04), (4, 0.025)]
PHI = {m: rng.uniform(0, 2 * np.pi) for m, _ in MODES}
KAP = {m: rng.uniform(-1, 1) for m, _ in MODES}
occ = None


def cavity_outline(theta, z):
    t = z / L
    r = 1.0 + sum(a * np.cos(m * theta + PHI[m] + np.pi * KAP[m] * t) for m, a in MODES)
    r *= 1.0 + 0.04 * np.sin(np.pi * t)
    return CAV_A * r * np.cos(theta), CAV_B * r * np.sin(theta)


def cavity_radius(theta, z):
    x, y = cavity_outline(theta, z)
    return np.hypot(x, y)


def vol(out):
    return [t for d, t in out if d == 3][0]


# ------------------------------------------------------------------ drain
def slot_centres():
    res = []
    for k in range(N_SLOTS):
        th = 2 * np.pi * k / N_SLOTS
        x, y = RING * A_D * np.cos(th), RING * B_D * np.sin(th)
        res.append((x, y, np.arctan2(y, x)))
    return res


def drain_solid():
    body = vol(occ.extrude([(2, occ.addDisk(0, 0, 0, A_D, B_D))], 0, 0, L))
    tools = []
    for x, y, ang in slot_centres():
        d = occ.addDisk(x, y, CH_Z0, SLOT_RX, SLOT_RY)
        occ.rotate([(2, d)], x, y, CH_Z0, 0, 0, 1, ang)
        tools.append((3, vol(occ.extrude([(2, d)], 0, 0, L - CH_Z0))))
        for z in HOLE_Z:
            tools.append((3, occ.addCylinder(x, y, z, 9 * np.cos(ang), 9 * np.sin(ang), 0, HOLE_D / 2)))
    core = occ.addDisk(0, 0, CH_Z0, CORE_R, CORE_R)
    tools.append((3, vol(occ.extrude([(2, core)], 0, 0, L - CH_Z0))))
    for z in CORE_HOLE_Z:                      # cross holes: core lumen -> slot 0 -> outside
        tools.append((3, occ.addCylinder(0, 0, z, 9, 0, 0, HOLE_D / 2)))
    out, _ = occ.cut([(3, body)], tools, removeObject=True, removeTool=True)
    return vol(out)


def dist_to_drain(p):
    """approximate distance (mm) from points p (n,3) to the drain envelope; negative inside"""
    th = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    E = np.stack([A_D * np.cos(th), B_D * np.sin(th)], 1)
    d = np.min(np.linalg.norm(p[:, None, :2] - E[None], axis=2), axis=1)
    inside = (p[:, 0] / A_D) ** 2 + (p[:, 1] / B_D) ** 2 < 1
    return np.where(inside, -d, d)


# ------------------------------------------------------------------ fragments
def rand_rot():
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def make_shape(kind):
    """local-frame description of a fragment: sections along local x.
    returns dict(secs=list of (n,3) point arrays, spheres=(m,4) [x,y,z,r], size=largest dimension)"""
    n_sec = 7
    if kind == "strand":
        length = rng.uniform(5.0, 8.0)                  # 6 to 10 hole diameters
        ry = rz = rng.uniform(0.28, 0.45)
        bend, noise = rng.uniform(0.5, 1.6), 0.18
    elif kind == "flake":
        length = rng.uniform(2.5, 5.5)
        ry, rz = rng.uniform(1.0, 2.0), rng.uniform(0.25, 0.38)
        bend, noise = rng.uniform(0.2, 0.7), 0.22
    else:                                              # blob
        length = rng.uniform(1.4, 3.2)
        ry = rz = rng.uniform(0.55, 1.1)
        bend, noise = rng.uniform(0.0, 0.3), 0.25
    ts = np.linspace(-0.5, 0.5, n_sec) * length
    ph = rng.uniform(0, 2 * np.pi, 3)
    phi = np.linspace(0, 2 * np.pi, 11)[:-1] + rng.uniform(0, 0.3)
    secs, sph = [], []
    for i, t in enumerate(ts):
        s = (i + 0.5) / n_sec
        w = 0.30 + 0.70 * np.sin(np.pi * s) ** 0.6       # blunt tapered ends
        rr = 1 + noise * (np.cos(2 * phi + ph[0] + 2 * s) * 0.6 + np.cos(3 * phi + ph[1]) * 0.4)
        cy = bend * np.sin(2 * np.pi * s * 0.8 + ph[2])
        cz = 0.6 * bend * np.cos(2 * np.pi * s * 0.7 + ph[0])
        pts = np.stack([np.full_like(phi, t), cy + w * ry * rr * np.cos(phi), cz + w * rz * rr * np.sin(phi)], 1)
        secs.append(pts)
        sph.append([t, cy, cz, 1.2 * w * max(ry, rz) * (1 + noise)])
    return dict(secs=secs, spheres=np.array(sph), size=length)


def place(shape, R, c):
    secs = [(p @ R.T) + c for p in shape["secs"]]
    sph = shape["spheres"].copy()
    sph[:, :3] = sph[:, :3] @ R.T + c
    return secs, sph


def valid(sph, placed):
    p, r = sph[:, :3], sph[:, 3]
    if np.any(p[:, 2] - r < 0.6) or np.any(p[:, 2] + r > L - 0.6):
        return False
    if np.any(dist_to_drain(p) - r < CLEAR):
        return False
    th = np.arctan2(p[:, 1], p[:, 0])
    rc = np.array([cavity_radius(t, z) for t, z in zip(th, p[:, 2])]) * 0.96
    if np.any(np.hypot(p[:, 0], p[:, 1]) + r + 0.6 > rc):
        return False
    for q in placed:
        d = np.linalg.norm(p[:, None, :] - q[None, :, :3], axis=2) - r[:, None] - q[None, :, 3]
        if np.any(d < CLEAR):
            return False
    return True


def to_gmsh(secs):
    wires = []
    for pts in secs:
        tags = [occ.addPoint(*q) for q in pts]
        tags.append(tags[0])
        wires.append(occ.addWire([occ.addSpline(tags)]))
    return vol(occ.addThruSections(wires, makeSolid=True, makeRuled=False))


def build_fragments(n_total=34, n_near_holes=7):
    kinds = (["strand"] * 11 + ["flake"] * 9 + ["blob"] * 9 + ["branched"] * 5)[:n_total]
    rng.shuffle(kinds)
    placed, vols, names = [], [], []
    holes = []
    for x, y, ang in slot_centres():
        for z in HOLE_Z:
            ex, ey = np.cos(ang) * 9, np.sin(ang) * 9
            # hole exit: where the hole axis crosses the ellipse envelope
            for s in np.linspace(0, 9, 900):
                px, py = x + s * np.cos(ang), y + s * np.sin(ang)
                if (px / A_D) ** 2 + (py / B_D) ** 2 >= 1:
                    holes.append((np.array([px, py, z]), np.array([np.cos(ang), np.sin(ang), 0.0])))
                    break
    near = set(rng.choice(len(kinds), n_near_holes, replace=False).tolist())
    hole_order = rng.permutation(len(holes))
    hi = 0
    for i, kind in enumerate(kinds):
        parts = [kind] if kind != "branched" else ["strand", "strand"]
        all_secs, all_sph, ok = [], [], False
        for attempt in range(400):
            shapes = [make_shape(k) for k in parts]
            R = rand_rot()
            if i in near and hi < len(holes):
                P0, nrm = holes[hole_order[hi]]
                sph_local = shapes[0]["spheres"]
                for s in np.arange(0.2, 6.0, 0.15):
                    c = P0 + nrm * (s + sph_local[:, 3].max())
                    secs, sph = place(shapes[0], R, c)
                    if valid(sph, placed):
                        ok = True
                        break
                if ok:
                    hi += 1
                    all_secs, all_sph = [secs], [sph]
                    break
                continue
            c = np.array([rng.uniform(-CAV_A, CAV_A), rng.uniform(-CAV_B, CAV_B), rng.uniform(1, L - 1)])
            if rng.random() > np.exp(-max(dist_to_drain(c[None])[0], 0) / 7.0):
                continue
            secs, sph = place(shapes[0], R, c)
            if not valid(sph, placed):
                continue
            all_secs, all_sph = [secs], [sph]
            if kind == "branched":
                # second strand starts near the middle of the first and leaves at an angle
                mid = (secs[3][:, :].mean(0))
                R2 = R @ np.array([[np.cos(0.9), -np.sin(0.9), 0], [np.sin(0.9), np.cos(0.9), 0], [0, 0, 1]])
                c2 = mid + R2 @ np.array([shapes[1]["size"] * 0.42, 0, 0])
                secs2, sph2 = place(shapes[1], R2, c2)
                if not valid(np.vstack([sph2]), placed + all_sph):
                    # parts overlap each other by design, so only test against others
                    pass
                all_secs.append(secs2)
                all_sph.append(sph2)
                if not valid(sph2, placed):
                    all_secs, all_sph = [], []
                    continue
            ok = True
            break
        if not ok:
            continue
        vs = [to_gmsh(s) for s in all_secs]
        if len(vs) == 2:
            try:
                out, _ = occ.fuse([(3, vs[0])], [(3, vs[1])])
                vs = [t for d, t in out if d == 3]
            except Exception:
                occ.remove([(3, vs[1])], recursive=True)
                vs = [vs[0]]
                kind = "strand"
        vols += vs
        names += [kind] * len(vs)
        placed += all_sph
    return vols, names


def cavity_volume():
    wires = []
    for z in np.linspace(0, L, 7):
        th = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        pts = [occ.addPoint(*cavity_outline(t, z), z) for t in th]
        pts.append(pts[0])
        wires.append(occ.addWire([occ.addSpline(pts)]))
    return vol(occ.addThruSections(wires, makeSolid=True, makeRuled=False))


def fresh(name):
    global occ
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ


def main(outdir):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    fresh("drain")
    d = drain_solid()
    occ.synchronize()
    gmsh.write(f"{outdir}/drain_perforated.step")
    fresh("frag")
    st = rng.bit_generator.state
    fv, names = build_fragments()
    occ.synchronize()
    gmsh.write(f"{outdir}/fragments_only.step")
    print("fragments:", {k: names.count(k) for k in set(names)}, "total", len(names))
    # assembly: rebuild with the same random state
    fresh("assembly")
    rng.bit_generator.state = st
    d = drain_solid()
    fv, names = build_fragments()
    cav = cavity_volume()
    occ.fragment([(3, cav)], [(3, d)] + [(3, v) for v in fv])
    occ.synchronize()
    gmsh.write(f"{outdir}/drain_fluid_fragments.step")
    print("assembly volumes:", len(gmsh.model.getEntities(3)))
    gmsh.finalize()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
