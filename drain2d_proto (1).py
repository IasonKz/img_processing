"""
2D prototype: compression of a multi-channel drain cross-section.

1. Mechanics: plane strain, compressible neo-Hookean, P1 triangles, Newton with
   energy line search. Loaded by two flat platens (prescribed vertical displacement).
   The channels are meshed too and treated as a very soft "pseudo-solid" so the
   fluid mesh follows the wall (same idea as ALE mesh motion in COMSOL).
2. Flow: fully developed laminar flow along the axis in each channel,
   laplace(w) = -1 on the DEFORMED channel cross-section, w = 0 on the wall,
   Q = integral of w dA (per unit pressure gradient / viscosity).

Units: mm, E_solid = 1 (results are normalised, so the stiffness scale cancels).
Geometry is a simplified analogue of the real design, not the real section.
"""
import time
import numpy as np
import gmsh
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------- parameters
P = dict(a=9.0, b=5.0,            # ellipse semi-axes (mm), body ~18 x 10 mm
         n_slots=8, slot_rx=1.4, slot_ry=0.45, ring=0.58, core_r=0.7,
         h=0.17,                  # mesh size (mm)
         platen_half=2.0,         # half-width of flat contact patch (mm)
         nu=0.40, fill_ratio=1e-3)  # Poisson ratio, soft-fill stiffness ratio


# ---------------------------------------------------------------- mesh
def build_mesh(p=P):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("drain")
    occ = gmsh.model.occ
    body = occ.addDisk(0, 0, 0, p["a"], p["b"])
    chans = []
    for k in range(p["n_slots"]):
        th = 2 * np.pi * k / p["n_slots"]
        x, y = p["ring"] * p["a"] * np.cos(th), p["ring"] * p["b"] * np.sin(th)
        d = occ.addDisk(x, y, 0, p["slot_rx"], p["slot_ry"])
        occ.rotate([(2, d)], x, y, 0, 0, 0, 1, np.arctan2(y, x))
        chans.append(d)
    chans.append(occ.addDisk(0, 0, 0, p["core_r"], p["core_r"]))
    occ.fragment([(2, body)], [(2, c) for c in chans])
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", p["h"])
    gmsh.option.setNumber("Mesh.MeshSizeMin", p["h"] * 0.5)
    gmsh.model.mesh.generate(2)

    ntags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    order = np.argsort(ntags)
    ntags, xyz = ntags[order], xyz[order][:, :2]
    idx = {int(t): i for i, t in enumerate(ntags)}
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    areas = {s: occ.getMass(2, s) for s in surfs}
    solid_tag = max(areas, key=areas.get)
    fluid_tags = sorted([s for s in surfs if s != solid_tag])
    tris, region = [], []
    for s in surfs:
        _, _, nt = gmsh.model.mesh.getElements(2, s)
        t = np.array([idx[int(v)] for v in nt[0]]).reshape(-1, 3)
        tris.append(t)
        region += [0 if s == solid_tag else 1 + fluid_tags.index(s)] * len(t)
    gmsh.finalize()
    return xyz, np.vstack(tris), np.array(region)


# ---------------------------------------------------------------- mechanics
def psi(F, mu, lam):
    J = jnp.linalg.det(F)
    lnJ = jnp.log(J)
    return 0.5 * mu * (jnp.sum(F * F) - 2.0) - mu * lnJ + 0.5 * lam * lnJ ** 2


def elem_energy(xe, Minv, A0, mu, lam):
    x = xe.reshape(3, 2)
    D = jnp.stack([x[1] - x[0], x[2] - x[0]], axis=1)
    return A0 * psi(D @ Minv, mu, lam)


e_fun = jax.jit(jax.vmap(elem_energy))
e_grad = jax.jit(jax.vmap(jax.grad(elem_energy)))
e_hess = jax.jit(jax.vmap(jax.hessian(elem_energy)))


class Model:
    def __init__(self, xyz, tris, region, p=P):
        self.X, self.T, self.reg, self.p = xyz, tris, region, p
        X0 = xyz[tris]                                     # (ne,3,2)
        M = np.stack([X0[:, 1] - X0[:, 0], X0[:, 2] - X0[:, 0]], axis=2)
        self.Minv = np.linalg.inv(M)
        self.A0 = 0.5 * np.abs(np.linalg.det(M))
        E = np.where(region == 0, 1.0, p["fill_ratio"])
        nu = p["nu"]
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.dofs = np.stack([2 * tris[:, [0, 0, 1, 1, 2, 2]]
                              + np.array([0, 1, 0, 1, 0, 1])], axis=0)[0]
        self.nd = 2 * len(xyz)
        r = (xyz[:, 0] / p["a"]) ** 2 + (xyz[:, 1] / p["b"]) ** 2
        on_bdry = r > 1 - 1e-4
        self.top = np.where(on_bdry & (xyz[:, 1] > 0) & (np.abs(xyz[:, 0]) <= p["platen_half"]))[0]
        self.bot = np.where(on_bdry & (xyz[:, 1] < 0) & (np.abs(xyz[:, 0]) <= p["platen_half"]))[0]
        self.xfix = np.array([self.top[np.argmin(np.abs(xyz[self.top, 0]))],
                              self.bot[np.argmin(np.abs(xyz[self.bot, 0]))]])

    def args(self, u):
        xe = (self.X + u.reshape(-1, 2))[self.T].reshape(-1, 6)
        return xe, self.Minv, self.A0, self.mu, self.lam

    def energy(self, u):
        return float(np.sum(e_fun(*self.args(u))))

    def resid_tangent(self, u):
        a = self.args(u)
        g = np.asarray(e_grad(*a))
        H = np.asarray(e_hess(*a))
        r = np.zeros(self.nd)
        np.add.at(r, self.dofs, g)
        rows = np.repeat(self.dofs[:, :, None], 6, axis=2).ravel()
        cols = np.repeat(self.dofs[:, None, :], 6, axis=1).ravel()
        K = sp.coo_matrix((H.ravel(), (rows, cols)), shape=(self.nd, self.nd)).tocsr()
        return r, K

    def solve_step(self, u, delta, tol=1e-8, maxit=30):
        u = u.copy()
        fixed = np.zeros(self.nd, bool)
        u[2 * self.top + 1] = -delta / 2
        u[2 * self.bot + 1] = +delta / 2
        fixed[2 * self.top + 1] = fixed[2 * self.bot + 1] = True
        fixed[2 * self.xfix] = True
        u[2 * self.xfix] = 0.0
        free = ~fixed
        E0 = self.energy(u)
        if not np.isfinite(E0):
            return u, False
        for _ in range(maxit):
            r, K = self.resid_tangent(u)
            if np.linalg.norm(r[free]) < tol:
                return u, True
            du = np.zeros(self.nd)
            du[free] = spla.spsolve(K[free][:, free].tocsc(), -r[free])
            a, ok = 1.0, False
            for _ in range(14):
                En = self.energy(u + a * du)
                if np.isfinite(En) and En < E0:
                    ok = True
                    break
                a *= 0.5
            if not ok:
                return u, False
            u, E0 = u + a * du, En
        return u, False


# ---------------------------------------------------------------- flow
def channel_flow(model, u):
    """Poisson problem on the deformed channel elements. Returns Q per channel,
    area per channel and the nodal field w (G = 1)."""
    ch = model.reg > 0
    tri = model.T[ch]
    x = (model.X + u.reshape(-1, 2))[tri]                     # (n,3,2)
    b = np.stack([x[:, 1, 1] - x[:, 2, 1], x[:, 2, 1] - x[:, 0, 1], x[:, 0, 1] - x[:, 1, 1]], 1)
    c = np.stack([x[:, 2, 0] - x[:, 1, 0], x[:, 0, 0] - x[:, 2, 0], x[:, 1, 0] - x[:, 0, 0]], 1)
    A = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * A[:, None, None])
    n = len(model.X)
    rows = np.repeat(tri[:, :, None], 3, axis=2).ravel()
    cols = np.repeat(tri[:, None, :], 3, axis=1).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    f = np.zeros(n)
    np.add.at(f, tri.ravel(), np.repeat(A / 3, 3))
    in_ch = np.zeros(n, bool); in_ch[tri.ravel()] = True
    in_sol = np.zeros(n, bool); in_sol[model.T[~ch].ravel()] = True
    free = in_ch & ~in_sol                                    # interior channel nodes
    w = np.zeros(n)
    w[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    reg = model.reg[ch]
    nch = reg.max()
    Q = np.array([np.sum(A[reg == k] * w[tri[reg == k]].mean(1)) for k in range(1, nch + 1)])
    Ar = np.array([np.sum(A[reg == k]) for k in range(1, nch + 1)])
    return Q, Ar, w


# ---------------------------------------------------------------- driver
def run(d_max_frac=0.45, verbose=True):
    t0 = time.time()
    xyz, tris, reg = build_mesh()
    m = Model(xyz, tris, reg)
    H = 2 * P["b"]
    u = np.zeros(m.nd)
    Q0, A0c, _ = channel_flow(m, u)
    if verbose:
        R = P["core_r"]
        print(f"mesh: {len(xyz)} nodes, {len(tris)} elements, {reg.max()} channels")
        print(f"check core channel (circle): Q_fem/Q_exact = {Q0[-1] / (np.pi * R ** 4 / 8):.4f}")
    hist = [dict(frac=0.0, Q=Q0.copy(), A=A0c.copy(), u=u.copy())]
    d, ds = 0.0, 0.1
    while d < d_max_frac * H - 1e-9:
        dn = min(d + ds, d_max_frac * H)
        un, ok = m.solve_step(u, dn)
        if ok:
            Q, Ar, _ = channel_flow(m, un)
            d, u = dn, un
            hist.append(dict(frac=d / H, Q=Q, A=Ar, u=u.copy()))
            ds = min(ds * 1.3, 0.2)
        else:
            ds *= 0.5
            if ds < 1e-3:
                if verbose:
                    print(f"stopped at compression {100 * d / H:.1f}% (no convergence)")
                break
    if verbose:
        print(f"{len(hist) - 1} load steps, {time.time() - t0:.1f} s")
    return m, hist, Q0


if __name__ == "__main__":
    m, hist, Q0 = run()
    for fr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        h = min(hist, key=lambda s: abs(s["frac"] - fr))
        print(f"compression {100 * h['frac']:4.1f}%  Q/Q0 = {h['Q'].sum() / Q0.sum():.3f}")
