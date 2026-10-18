"""Figures, animations and per-problem tables for the HTML benchmark report.

Usage (from the project folder, same environment as thermo_benchmark.py):
    python report/make_assets.py --part method,kf,ik,mpc,heat,rand,tables
Each part can run in its own process. Outputs go to report/assets/.
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
from thermox.sampler import sample_identity_diffusion
from thermox.utils import ProcessedDriftMatrix

import kf_thermox_pick_place as E
import thermo_benchmark as T
import thermo_benchmark_report as R

OUT = os.path.join(HERE, "assets")
RES = R.load(os.path.join(ROOT, "results_bench"))
TGT = 1e-4
INK, INK2, MUTED, GRID, AXIS, SURF = R.INK, R.INK2, R.MUTED, R.GRID, R.AXIS, R.SURF
C = dict(exact=INK, plain=R.COL["plain"], ir_cold=R.COL["ir_cold"], ir_warm=R.COL["ir_warm"],
         cg=R.COL["cg"])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "legend.fontsize": 8})


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name, dpi=130):
    fig.savefig(os.path.join(OUT, name), dpi=dpi)
    plt.close(fig)
    print("wrote", name, flush=True)


def fig_frame(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, name, fps=6):
    pil = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
           for f in frames]
    pil[0].save(os.path.join(OUT, name), save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote", name, flush=True)


# ============================================================ solver used in loops
class BenchSolver:
    """The benchmark's device model and IR loop (thermo_benchmark.run_thermo) as a
    callable with warm-start state, for closed-loop runs. rounds_max=1 + target=0 + cold
    = plain thermox."""

    def __init__(self, N, target=TGT, warm=True, jacobi=True, rounds_max=T.MAX_ROUNDS,
                 seed=0, stall=True):
        self.N, self.target, self.warm, self.jacobi = N, target, warm, jacobi
        self.rounds_max, self.stall = rounds_max, stall
        self.dev = T.Device(N, seed)
        self.X, self.A_prog = None, None
        self.rounds, self.errs, self.trace = [], [], []

    def program(self, A):
        if A is self.A_prog:
            return
        self.P = 1 / np.sqrt(np.diag(A)) if self.jacobi else np.ones(A.shape[0])
        As = self.P[:, None] * A * self.P[None, :]
        self.c = np.linalg.eigvalsh(As)[-1]
        self.stable = self.dev.program(As / self.c)
        self.A_prog = A

    def __call__(self, A, B):
        vec = B.ndim == 1
        B2 = B[:, None] if vec else B
        self.program(A)
        Xs = np.linalg.solve(A, B2)
        nx = np.linalg.norm(Xs)
        X = self.X.copy() if (self.warm and self.X is not None
                              and self.X.shape == B2.shape) else np.zeros_like(B2)
        err = (lambda Z: np.linalg.norm(Z - Xs) / nx) if nx > 0 else (lambda Z: 0.0)
        errs = [err(X)]
        while self.stable and len(errs) <= self.rounds_max and errs[-1] > self.target:
            Rr = self.P[:, None] * (B2 - A @ X) / self.c
            s = np.max(np.abs(Rr), axis=0)
            s = np.where(s > 0, s, 1.0)
            X = X + self.P[:, None] * (self.dev.solve(Rr / s) * s)
            errs.append(err(X))
            if self.stall and len(errs) > 6 and errs[-1] > 0.5 * errs[-6]:
                break
        self.X = X
        self.rounds.append(len(errs) - 1)
        self.errs.append(errs[-1])
        self.trace = errs
        return X[:, 0] if vec else X


def best_cfg(name):
    """Configuration the report shows as 'IR': best warm (sequences) or cold, at 1e-4."""
    r = RES[name]
    b = (R.best_ir(r, TGT, True) if r["seq"] else None) or R.best_ir(r, TGT, False)
    return b


def ir_solver(name, seed=0, **kw):
    b = best_cfg(name)
    if b is None:                                 # never converges: show N=1000 warm
        return BenchSolver(1000, warm=True, jacobi=False, seed=seed, **kw), None
    return BenchSolver(b["N"], warm=b["warm"], jacobi=b["prec"] == "jac", seed=seed, **kw), b


def plain_solver(name, N, seed=0):
    b = best_cfg(name)
    jac = b["prec"] == "jac" if b else False
    return BenchSolver(int(N), target=0.0, warm=False, jacobi=jac, rounds_max=1, seed=seed)


def equal_budget(name):
    b = best_cfg(name)
    return int(round(b["samples"])) if b else 40 * 1000


# ============================================================ convergence curves
def cg_trace(A, B, X0, jacobi, maxit, tol=1e-10):
    Xs = np.linalg.solve(A, B)
    nx = np.linalg.norm(Xs)
    X = X0.copy()
    Minv = (1.0 / np.diag(A))[:, None] if jacobi else np.ones((A.shape[0], 1))
    Rr = B - A @ X
    Z = Minv * Rr
    P = Z.copy()
    rz = np.sum(Rr * Z, axis=0)
    errs = [np.linalg.norm(X - Xs) / nx]
    for _ in range(maxit):
        if errs[-1] < tol:
            break
        AP = A @ P
        pap = np.sum(P * AP, axis=0)
        alpha = np.divide(rz, pap, out=np.zeros_like(rz), where=pap > 0)
        X = X + alpha * P
        Rr = Rr - alpha * AP
        Z = Minv * Rr
        rzn = np.sum(Rr * Z, axis=0)
        beta = np.divide(rzn, rz, out=np.zeros_like(rz), where=rz > 0)
        P = Z + beta * P
        rz = rzn
        errs.append(np.linalg.norm(X - Xs) / nx)
    return np.array(errs)


def _median_traces(traces, n):
    M = np.full((len(traces), n), np.nan)
    for i, t in enumerate(traces):
        M[i, :min(n, len(t))] = t[:n]
    return np.nanmedian(M, axis=0)


def convergence(name, idx=None, seeds=5):
    """Error curves for one representative system of a problem set."""
    systems = T.PROBLEMS[name]["build"]()
    idx = len(systems) // 2 if idx is None else idx
    A, B = systems[idx][0], systems[idx][1]
    seq = RES[name]["seq"]
    X0w = np.linalg.solve(*systems[idx - 1][:2]) if seq and idx > 0 else None
    b = best_cfg(name)
    jac = b["prec"] == "jac" if b else False
    N_ir = b["N"] if b else 1000
    cold_cfg = R.best_ir(RES[name], TGT, False)
    N_cold = cold_cfg["N"] if cold_cfg else N_ir
    out = dict(idx=idx, N_ir=N_ir, N_cold=N_cold, jac=jac)
    Ns = [10, 30, 100, 300, 1000, 3000, 10000]
    out["plain_N"] = Ns
    out["plain"] = [float(np.median([np.linalg.norm(
        BenchSolver(N, 0.0, False, jac, 1, seed=s)(A, B) - np.linalg.solve(A, B)) /
        np.linalg.norm(np.linalg.solve(A, B)) for s in range(seeds)])) for N in Ns]

    def ir_runs(N, X0):
        tr = []
        for s in range(seeds):
            bs = BenchSolver(N, target=1e-9, warm=True, jacobi=jac, rounds_max=40, seed=s,
                             stall=False)
            bs.X = None if X0 is None else X0.copy()
            bs(A, B)
            tr.append(bs.trace)
        return _median_traces(tr, 41)

    out["ir_cold"] = ir_runs(N_cold, None).tolist()
    out["ir_warm"] = ir_runs(N_ir, X0w).tolist() if seq else None
    d = A.shape[0]
    cg = [cg_trace(A, B, np.zeros_like(B), j, 4 * d + 20) for j in (False, True)]
    out["cg_cold"] = min(cg, key=lambda e: np.argmax(e < TGT) if np.any(e < TGT) else 1e9
                         ).tolist()
    if seq:
        cgw = [cg_trace(A, B, X0w, j, 4 * d + 20) for j in (False, True)]
        out["cg_warm"] = min(cgw, key=lambda e: np.argmax(e < TGT) if np.any(e < TGT)
                             else 1e9).tolist()
    return out


def fig_convergence(name, cv):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    a1.loglog(cv["plain_N"], cv["plain"], "-o", color=C["plain"], lw=2, ms=5, mec=SURF,
              label="plain thermox (one run of N samples)")
    rc = np.array(cv["ir_cold"])
    r = np.arange(len(rc))
    a1.loglog(np.maximum(r, 0.5) * cv["N_cold"], rc, "-o", color=C["ir_cold"], lw=2, ms=4,
              mec=SURF, label=f"IR cold, N={cv['N_cold']} per round")
    if cv["ir_warm"] is not None:
        rw = np.array(cv["ir_warm"])
        a1.loglog(np.maximum(np.arange(len(rw)), 0.5) * cv["N_ir"], rw, "-o",
                  color=C["ir_warm"], lw=2, ms=4, mec=SURF,
                  label=f"IR warm, N={cv['N_ir']} per round")
    a1.axhline(TGT, color=AXIS, lw=1)
    a1.text(12, TGT * 1.4, "target 1e-4", color=MUTED, fontsize=8)
    a1.set_xlabel("device samples (fastest relaxation times)")
    a1.set_ylabel("relative error")
    a1.set_title("Chip time: plain averaging vs refinement", loc="left", color=INK)
    a1.set_ylim(1e-9, 20)
    a1.legend(loc="lower left")
    style(a1)
    cgc = np.array(cv["cg_cold"])
    a2.semilogy(np.arange(len(cgc)), cgc, "-", color=C["cg"], lw=2, label="digital CG, cold")
    if cv.get("cg_warm"):
        cgw = np.array(cv["cg_warm"])
        a2.semilogy(np.arange(len(cgw)), cgw, "-", color=C["cg"], lw=1.2, alpha=0.55,
                    label="digital CG, warm")
    a2.semilogy(np.arange(len(rc)), rc, "-o", color=C["ir_cold"], lw=2, ms=4, mec=SURF,
                label="IR cold (1 residual per round)")
    if cv["ir_warm"] is not None:
        a2.semilogy(np.arange(len(rw)), rw, "-o", color=C["ir_warm"], lw=2, ms=4, mec=SURF,
                    label="IR warm")
    a2.axhline(TGT, color=AXIS, lw=1)
    a2.set_xlabel("digital matrix-vector products")
    a2.set_ylabel("relative error")
    a2.set_ylim(1e-9, 20)
    a2.set_xlim(0, max(10, min(len(cgc), 60)))
    a2.set_title("Digital work: CG iterations vs IR rounds", loc="left", color=INK)
    a2.legend(loc="upper right")
    style(a2)
    fig.tight_layout()
    save(fig, f"conv_{name}.png")


# ============================================================ tables
def tables():
    out = {}

    def f(v):
        if v is None or not np.isfinite(v):
            return "never"
        for scale, unit, lim in ((1e6, "µs", 1e-3), (1e3, "ms", 1), (1, "s", 60),
                                 (1 / 60, "min", 3600), (1 / 3600, "h", 86400)):
            if v < lim:
                return f"{v * scale:.3g} {unit}"
        return f"{v / 86400:.3g} days"
    for name in R.ROBOT + R.CLASSIC + R.RANDOM:
        r = RES[name]
        Np, ext = R.plain_needed(r, TGT)
        bc, bw = R.best_ir(r, TGT, False), R.best_ir(r, TGT, True) if r["seq"] else None
        cg = R.best_cg(r, TGT)
        t4, t2 = R.wall_clock(r, TGT), R.wall_clock(r, 1e-2)

        def cell(b):
            return ("stalls", "–") if b is None else \
                (f"{b['samples']:,.0f}", f"{b['rounds']:.1f} (N={b['N']})")
        c_s, c_r = cell(bc)
        w_s, w_r = cell(bw) if r["seq"] else ("not a sequence", "–")
        p = f"{Np:.1e}".replace("e+0", "e").replace("e+", "e") + (" *" if ext else "")
        cost = (
            "<table class='num'><thead><tr><th></th><th>Plain thermox</th><th>IR cold</th>"
            "<th>IR warm</th><th>Digital CG</th></tr></thead><tbody>"
            f"<tr><th>Device samples to 1e-4</th><td>{p}</td><td>{c_s}</td><td>{w_s}</td>"
            "<td>–</td></tr>"
            f"<tr><th>Digital mat-vec products</th><td>–</td><td>{c_r}</td><td>{w_r}</td>"
            f"<td>{cg['iters']:.1f} ({'warm ' if cg['warm'] else ''}{cg['kind'].upper()})</td>"
            "</tr></tbody></table>")
        star = lambda t: " *" if t["plain_extrap"] else ""
        rows = [("CPU Cholesky", "chol", False), ("CPU CG", "cg", False)] + \
            [(f"Plain thermox, {lab}", "plain_" + hw, True) for hw, lab in
             (("proto", "prototype chip"), ("chip", "optimistic chip"),
              ("limit", "physical limit"))] + \
            [(f"Thermox + IR, {lab}", hw, False) for hw, lab in
             (("proto", "prototype chip"), ("chip", "optimistic chip"),
              ("limit", "physical limit"))]
        time = (
            "<table class='num'><thead><tr><th>Time per solve</th><th>to 1e-2</th>"
            "<th>to 1e-4</th></tr></thead><tbody>" +
            "".join(f"<tr class='{'r-plain' if k.startswith('plain') else ''}'><th>{lab}</th>" +
                    "".join(f"<td>{f(t[k])}{star(t) if is_plain else ''}</td>"
                            for t in (t2, t4)) + "</tr>" for lab, k, is_plain in rows) +
            "</tbody></table>" +
            ("<p class='tnote'>* Plain thermox did not reach this target within 10,000 samples; "
             "its time uses the samples the 1/√N trend says it would need.</p>"
             if t2["plain_extrap"] or t4["plain_extrap"] else ""))
        cpu = min(t4["chol"], t4["cg"])
        if np.isfinite(t4["chip"]):
            ratio = t4["chip"] / cpu
            fx = lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"
            speed = (f"<span class='pill bad'>optimistic chip {fx(ratio)}× slower than CPU"
                     "</span>" if ratio >= 1.15 else
                     f"<span class='pill ok'>optimistic chip {fx(1 / ratio)}× faster than CPU"
                     "</span>" if ratio <= 1 / 1.15 else
                     "<span class='pill'>optimistic chip about level with CPU</span>")
        else:
            speed = "<span class='pill bad'>no thermo time: never converges</span>"
        conv = ("<span class='pill ok'>IR reaches 1e-4</span>" if (bc or bw) else
                "<span class='pill bad'>IR stalls before 1e-4</span>")
        out[name] = dict(cost=cost, time=time, verdict=conv + speed, d=r["d"], m=r["m"],
                         cond=r["cond_median"], n=r["n_systems"], n_mat=r["n_matrices"],
                         jac_cond=r["jacobi_cond_median"], seq=r["seq"],
                         t_chol=t4["chol"], t_cg=t4["cg"], t_chip=t4["chip"],
                         t_proto=t4["proto"], t_limit=t4["limit"])
    with open(os.path.join(OUT, "tables.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote tables.json")


# ============================================================ methods
def _ou_2d():
    th = np.deg2rad(30)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = Q @ np.diag([1.0, 0.25]) @ Q.T
    xs = np.array([1.2, 0.8])
    return A, A @ xs, xs


def part_method():
    A, b, xs = _ou_2d()
    w, V = np.linalg.eigh(A)
    proc = ProcessedDriftMatrix(jnp.asarray(A), jnp.asarray(w), jnp.asarray(V),
                                jnp.asarray(V.T), jnp.asarray(w), jnp.asarray(V))
    dt, n = 0.05, 1600
    ts = jnp.arange(n + 1) * dt
    path = np.asarray(sample_identity_diffusion(jax.random.PRNGKey(3), ts, jnp.zeros(2),
                                                proc, jnp.asarray(xs)))
    run_mean = np.cumsum(path, axis=0) / np.arange(1, n + 2)[:, None]
    err = np.linalg.norm(run_mean - xs, axis=1) / np.linalg.norm(xs)
    frames = []
    for k in list(range(8, 200, 8)) + list(range(200, n + 1, 40)):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=95)
        a1.plot(path[:k + 1, 0], path[:k + 1, 1], color=C["plain"], lw=0.6, alpha=0.35)
        a1.plot(*path[k], "o", color=C["plain"], ms=6, mec=SURF)
        a1.plot(*run_mean[k], "o", color=INK, ms=8, mec=SURF, label="running average")
        a1.plot(*xs, "+", color=INK, ms=16, mew=2, label="true solution x*")
        a1.plot(0, 0, "s", color=MUTED, ms=6, label="start x₀ = 0")
        a1.set_xlim(-1.6, 3.6)
        a1.set_ylim(-1.6, 3.2)
        a1.set_aspect("equal")
        a1.set_title(f"Chip state wandering by heat noise, t = {k * dt:4.1f} τ", loc="left",
                     color=INK)
        a1.legend(loc="upper left")
        style(a1)
        a2.loglog(np.arange(1, k + 2) * dt, err[:k + 1], color=INK, lw=2)
        tt = np.logspace(0, np.log10(n * dt), 20)
        a2.loglog(tt, 0.9 / np.sqrt(tt), color=MUTED, lw=1)
        a2.text(30, 0.9 / np.sqrt(30) * 1.3, "∝ 1/√t", color=MUTED, fontsize=8)
        a2.set_xlim(dt, n * dt * 1.1)
        a2.set_ylim(5e-3, 2)
        a2.set_xlabel("averaging time (relaxation times τ)")
        a2.set_ylabel("error of the running average")
        a2.set_title("Error shrinks only like 1/√time", loc="left", color=INK)
        style(a2)
        fig.tight_layout()
        frames.append(fig_frame(fig))
        plt.close(fig)
    save_gif(frames, "method_ou.gif", fps=8)

    # plain vs IR clouds at the same budget (30 samples), plus error vs budget
    A_s = A / np.linalg.eigvalsh(A)[-1]
    reps = 300
    plain, ir = [], []
    dev_p = T.Device(30, 1)
    dev_p.program(A_s)
    for s in range(reps // 50):
        bb = np.tile((A_s @ xs)[:, None], (1, 50))
        sc = np.max(np.abs(bb), axis=0)
        plain.append((dev_p.solve(bb / sc) * sc).T)
    plain = np.concatenate(plain)
    for s in range(reps):
        bs = BenchSolver(10, target=0.0, warm=False, jacobi=False, rounds_max=3, seed=s,
                         stall=False)
        ir.append(bs(A_s, A_s @ xs))
    ir = np.array(ir)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    for a, pts, col, lab in ((ax[0], plain, C["plain"], "plain thermox, N = 30"),
                             (ax[1], ir, C["ir_cold"], "IR, 3 rounds × N = 10")):
        a.plot(pts[:, 0], pts[:, 1], "o", color=col, ms=3, alpha=0.45, mec="none")
        a.plot(*xs, "+", color=INK, ms=16, mew=2)
        e = np.median(np.linalg.norm(pts - xs, axis=1)) / np.linalg.norm(xs)
        a.set_title(f"{lab}: median error {100 * e:.0f}%", loc="left", color=INK)
        a.set_xlim(-0.6, 3.0)
        a.set_ylim(-0.6, 2.2)
        a.set_aspect("equal")
        style(a)
    budgets = [10, 30, 100, 300, 1000, 3000]
    ep, ei = [], []
    for Bud in budgets:
        ep.append(np.median([np.linalg.norm(BenchSolver(Bud, 0.0, False, False, 1, seed=s)(
            A_s, A_s @ xs) - xs) / np.linalg.norm(xs) for s in range(40)]))
        ei.append(np.median([np.linalg.norm(BenchSolver(10, 0.0, False, False, Bud // 10,
                                                        seed=s, stall=False)(A_s, A_s @ xs)
                                            - xs) / np.linalg.norm(xs) for s in range(40)]))
    ax[2].loglog(budgets, ep, "-o", color=C["plain"], lw=2, ms=5, mec=SURF,
                 label="plain thermox, N = budget")
    ax[2].loglog(budgets, np.maximum(ei, 1e-16), "-o", color=C["ir_cold"], lw=2, ms=5,
                 mec=SURF, label="IR, rounds of N = 10")
    ax[2].set_xlabel("total device samples")
    ax[2].set_ylabel("median relative error")
    ax[2].set_title("Same chip time, very different accuracy", loc="left", color=INK)
    ax[2].legend(loc="lower left")
    style(ax[2])
    fig.tight_layout()
    save(fig, "method_ir_clouds.png")


# ============================================================ Kalman filter problems
def kf_run(ratio, solver, seed=0):
    p_true, _ = E.truth_and_noise(seed)
    Rc = T._camera_R(ratio)
    rng = np.random.default_rng(10_000 + seed)
    meas = p_true + rng.normal(size=p_true.shape) @ np.linalg.cholesky(Rc).T
    kf = E.KalmanCV(solver, meas[0])
    kf.R = Rc
    n = int(round(E.T_GRASP / E.CTRL_DT)) + 1
    xh, vh, mu = (np.full((n, 3), np.nan) for _ in range(3))
    for k in range(n):
        if k > 0:
            kf.predict(E.CTRL_DT)
        if k % E.CAM_EVERY == 0 and k > 0:
            kf.update(meas[k])
            mu[k] = meas[k]
        xh[k], vh[k] = kf.x[:3], kf.x[3:]
    tv = np.gradient(p_true, E.CTRL_DT, axis=0)[:n]
    return dict(t=np.arange(n) * E.CTRL_DT, p=p_true[:n], v=tv, xh=xh, vh=vh, meas=mu,
                gain=np.array(kf.gain_err))


def part_kf():
    # illustration: camera noise shapes
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
    rng = np.random.default_rng(1)
    for j, ratio in enumerate((1, 3, 10)):
        Rc = T._camera_R(ratio)
        pts = rng.normal(size=(500, 3)) @ np.linalg.cholesky(Rc).T * 100
        for i, (a, b_, lab) in enumerate(((0, 1, "top view: x / y"),
                                          (0, 2, "side view: x / z"))):
            a_ = ax[i, j]
            a_.plot(pts[:, a], pts[:, b_], "o", color=C["plain"], ms=2.5, alpha=0.35,
                    mec="none")
            Sub = Rc[np.ix_([a, b_], [a, b_])] * 1e4
            w, V = np.linalg.eigh(Sub)
            ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
            a_.add_patch(Ellipse((0, 0), 4 * np.sqrt(w[-1]), 4 * np.sqrt(w[0]), angle=ang,
                                 fill=False, color=INK, lw=1.5))
            a_.set_xlim(-12, 12)
            a_.set_ylim(-12, 12)
            a_.set_aspect("equal")
            a_.set_xlabel(f"{'xyz'[a]} error [cm]")
            a_.set_ylabel(f"{'xyz'[b_]} error [cm]")
            style(a_)
            if i == 0:
                k = np.linalg.cond(Rc)
                a_.set_title(["Isotropic camera", "Depth noise 3×", "Depth noise 10×"][j] +
                             f"  (cond R = {k:.0f})", loc="left", color=INK)
            a_.text(-11.5, 10, lab, color=MUTED, fontsize=8)
    fig.suptitle("500 simulated camera measurements of a still cube; ellipse = 2σ. Deeper "
                 "noise along the viewing ray makes S = HPHᵀ + R ill-conditioned.",
                 color=INK2, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "kf_noise.png")
    data = {}
    for name, ratio in (("kf_iso", 1.0), ("kf_depth3", 3.0), ("kf_depth10", 10.0)):
        Nq = equal_budget(name)
        runs = {"exact": kf_run(ratio, E.ExactSolver()),
                "plain": kf_run(ratio, plain_solver(name, Nq))}
        irs, b = ir_solver(name)
        runs["ir"] = kf_run(ratio, irs)
        fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
        L = runs["exact"]
        m = ~np.isnan(L["meas"][:, 0])
        a = ax[0, 0]
        a.plot(L["meas"][m, 0], L["meas"][m, 1], "o", color=MUTED, ms=2.5, alpha=0.5,
               label="camera measurements")
        a.plot(L["p"][:, 0], L["p"][:, 1], color=INK, lw=2.2, label="true cube path")
        for key, col, lw, lab in (("plain", C["plain"], 1.6, f"plain thermox, N={Nq}"),
                                  ("exact", AXIS, 4.5, "exact solve"),
                                  ("ir", C["ir_warm"], 1.6, "thermox + IR")):
            a.plot(runs[key]["xh"][:, 0], runs[key]["xh"][:, 1], color=col, lw=lw, label=lab,
                   zorder=2 if key != "exact" else 1)
        a.set_xlim(L["p"][:, 0].min() - 0.06, L["p"][:, 0].max() + 0.06)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m] (belt direction)")
        a.set_title("Top view: cube path and Kalman estimates", loc="left", color=INK)
        fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=8.5)
        style(a)
        a = ax[0, 1]
        tu = np.arange(1, len(runs["ir"]["gain"]) + 1) * E.CAM_EVERY * E.CTRL_DT
        a.semilogy(tu, runs["plain"]["gain"], color=C["plain"], lw=1.4,
                   label=f"plain thermox, N={Nq}")
        a.semilogy(tu, np.maximum(runs["ir"]["gain"], 1e-9), color=C["ir_warm"], lw=1.6,
                   label="thermox + IR")
        a.axhline(TGT, color=AXIS, lw=1)
        a.set_xlabel("time [s]")
        a.set_ylabel("‖K − K_exact‖ / ‖K_exact‖")
        a.set_title("Kalman-gain error at each camera update", loc="left", color=INK)
        a.legend(loc="center right")
        style(a)
        for a, key, unit, lab in ((ax[1, 0], ("xh", "p"), "mm", "position"),
                                  (ax[1, 1], ("vh", "v"), "mm/s", "velocity")):
            for k2, col, lw in (("exact", AXIS, 4), ("plain", C["plain"], 1.3),
                                ("ir", C["ir_warm"], 1.3)):
                e = np.linalg.norm(runs[k2][key[0]] - runs[k2][key[1]], axis=1) * 1e3
                a.plot(runs[k2]["t"], e, color=col, lw=lw,
                       label={"exact": "exact solve", "plain": "plain thermox",
                              "ir": "thermox + IR"}[k2])
            a.set_xlabel("time [s]")
            a.set_ylabel(f"|error| [{unit}]")
            a.set_title(f"Estimated {lab} error", loc="left", color=INK)
            a.legend(loc="upper right")
            style(a)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"solve_{name}.png")
        rm = lambda L_, k1, k2: float(np.sqrt(np.nanmean(np.sum(
            (L_[k1][50:] - L_[k2][50:]) ** 2, axis=1))) * 1e3)
        data[name] = dict(N_eq=Nq, cfg=b, **{
            f"{k}_{q}": rm(runs[k], *pair) for k in runs for q, pair in
            (("pos", ("xh", "p")), ("vel", ("vh", "v")))},
            plain_gain=float(np.mean(runs["plain"]["gain"])),
            ir_gain=float(np.mean(runs["ir"]["gain"])),
            ir_rounds=float(np.mean(irs.rounds)))
        fig_convergence(name, convergence(name))
    with open(os.path.join(OUT, "data_kf.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=float)
    # a still of the pick-and-place scene from the existing robot video
    rdr = imageio.get_reader(os.path.join(ROOT, "results_ir", "pick_place_comparison.mp4"))
    for i, fr in enumerate(rdr):
        if i == 40:
            Image.fromarray(fr[:, :400]).save(os.path.join(OUT, "scene_still.png"))
            break
    print("wrote scene_still.png")


# ============================================================ inverse kinematics
def ik_run(solver, color, renderer):
    res = dict(pos=[], rot=[], lin=[], lin_t=[])

    class SolverIK(E.IK):
        def solve(self, q, p_des, R_des, iters=4, lam=0.05):
            d = self.d
            d.qpos[:7] = q
            for _ in range(iters):
                mujoco.mj_kinematics(self.m, d)
                mujoco.mj_comPos(self.m, d)
                Rm = d.site_xmat[self.sid].reshape(3, 3)
                ep = p_des - d.site_xpos[self.sid]
                Re = R_des @ Rm.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0],
                                     Re[1, 0] - Re[0, 1]])
                mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.sid)
                J = np.vstack([self.jp[:, :7], self.jr[:, :7]])
                A = J @ J.T + lam**2 * np.eye(6)
                e = np.r_[ep, er]
                y = solver(A, e)
                ye = np.linalg.solve(A, e)
                if np.linalg.norm(ye) > 0:
                    res["lin"].append(np.linalg.norm(y - ye) / np.linalg.norm(ye))
                    res["lin_t"].append(len(res["pos"]) * E.CTRL_DT)
                d.qpos[:7] = np.clip(d.qpos[:7] + J.T @ y, self.lo, self.hi)
            mujoco.mj_kinematics(self.m, d)
            res["pos"].append(np.linalg.norm(p_des - d.site_xpos[self.sid]))
            Re = R_des @ d.site_xmat[self.sid].reshape(3, 3).T
            res["rot"].append(np.degrees(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))))
            return d.qpos[:7].copy()

    cb, frames = E.make_recorder(renderer, color, every=8)
    orig, E.IK = E.IK, SolverIK
    try:
        met, L = E.run_trial(renderer._model, E.ExactSolver(), 0, frame_cb=cb)
    finally:
        E.IK = orig
    return met, L, {k: np.array(v) for k, v in res.items()}, frames


def part_ik():
    model = E.build_model()
    renderer = mujoco.Renderer(model, 330, 400, max_geom=6000)
    renderer._model = model
    Nq = equal_budget("ik")
    irs, b = ir_solver("ik")
    runs, panels = {}, []
    for key, solver, col, title in (
            ("exact", E.ExactSolver(), (0.2, 0.55, 1.0), "IK with exact solve"),
            ("plain", plain_solver("ik", Nq), (0.92, 0.41, 0.2), f"IK with plain thermox N={Nq}"),
            ("ir", irs, (0.93, 0.63, 0.0), "IK with thermox + IR")):
        met, L, res, frames = ik_run(solver, col, renderer)
        runs[key] = (met, L, res)
        panels.append(dict(frames=frames, title=title, color=col, grasp_err=met["grasp_err"],
                           every=8))
        print(key, met["grasp_err"], flush=True)
    E.compose_video(panels, os.path.join(OUT, "ik_robot.gif"))
    os.remove(os.path.join(OUT, "ik_robot.gif"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    t = np.arange(len(runs["exact"][2]["pos"])) * E.CTRL_DT
    for key, col, lw, lab in (("exact", AXIS, 3.5, "exact solve"),
                              ("plain", C["plain"], 1.3, f"plain thermox N={Nq}"),
                              ("ir", C["ir_warm"], 1.3, "thermox + IR")):
        res = runs[key][2]
        ax[0].semilogy(t, np.maximum(res["pos"], 1e-10) * 1e3, color=col, lw=lw, label=lab)
        ax[1].semilogy(t, np.maximum(res["rot"], 1e-8), color=col, lw=lw)
        if key != "exact":
            ax[2].semilogy(res["lin_t"], np.maximum(res["lin"], 1e-9), color=col, lw=0.8)
    for a, yl, ti in ((ax[0], "mm", "IK position residual after 4 iterations"),
                      (ax[1], "degrees", "IK orientation residual"),
                      (ax[2], "relative error", "Error of each IK linear solve")):
        a.set_xlabel("time [s]")
        a.set_ylabel(yl)
        a.set_title(ti, loc="left", color=INK)
        style(a)
    ax[2].axhline(TGT, color=AXIS, lw=1)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "solve_ik.png")
    # still for the problem illustration
    rdr = imageio.get_reader(os.path.join(OUT, "ik_robot.mp4"))
    for i, fr in enumerate(rdr):
        if i == 30:
            Image.fromarray(fr[:, :400]).save(os.path.join(OUT, "ik_still.png"))
            break
    k0, k1 = int(E.T_TRACK / E.CTRL_DT), int(E.T_GRASP / E.CTRL_DT)
    data = dict(N_eq=Nq, cfg=b, **{
        f"{k}_{q}": float(np.median(runs[k][2][q][k0:k1]) * (1e3 if q == "pos" else 1))
        for k in runs for q in ("pos", "rot")},
        **{f"{k}_grasp_mm": runs[k][0]["grasp_err"] * 1e3 for k in runs},
        **{f"{k}_success": runs[k][0]["success"] for k in runs},
        plain_lin=float(np.median(runs["plain"][2]["lin"])),
        ir_lin=float(np.median(runs["ir"][2]["lin"])), ir_rounds=float(np.mean(irs.rounds)))
    with open(os.path.join(OUT, "data_ik.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=float)
    fig_convergence("ik", convergence("ik"))


# ============================================================ MPC
def part_mpc():
    _, log = T.robot_log()
    data = {}
    for name, wr in (("mpc", 1e-2), ("mpc_hard", 1e-3)):
        systems = T.mpc_systems(wr, parts=True)
        Nq = equal_budget(name)
        irs, b = ir_solver(name)
        ticks = [8, len(systems) // 2, len(systems) - 12]
        sols = {"exact": {}, "plain": {}, "ir": {}}
        pl = plain_solver(name, Nq)
        for i, (A, B, info) in enumerate(systems):
            U = irs(A, B)
            if i in ticks:
                sols["ir"][i] = U[:, 0]
                sols["exact"][i] = np.linalg.solve(A, B)[:, 0]
                sols["plain"][i] = pl(A, B)[:, 0]
        H, n, dt = T.MPC_H, 7, T.MPC_STRIDE * E.CTRL_DT
        if name == "mpc":
            # illustration: receding horizon, Hessian structure, spectra
            A, B, info = systems[ticks[1]]
            k = info["k"]
            X = info["Phi"] @ info["x0"] + info["Gam"] @ sols["exact"][ticks[1]]
            q_pred = X.reshape(H, 2 * n)[:, :n]
            j = int(np.argmax(np.ptp(log["qdes"][k:k + 30], axis=0)))
            fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
            tt = np.arange(len(log["q"])) * E.CTRL_DT
            ax[0].plot(tt[k - 40:k + 1], log["q"][k - 40:k + 1, j], color=INK, lw=2,
                       label="joint angle so far")
            ax[0].plot(tt[k:k + 2 * H + 1], log["qdes"][k:k + 2 * H + 1, j], color=MUTED,
                       lw=1.5, label="reference ahead")
            ax[0].plot(tt[k] + dt * np.arange(1, H + 1), q_pred[:, j], "o-", color=C["cg"],
                       lw=2, ms=5, mec=SURF, label="MPC prediction (10 steps)")
            ax[0].axvline(tt[k], color=AXIS, lw=1)
            ax[0].set_xlabel("time [s]")
            ax[0].set_ylabel(f"joint {j + 1} angle [rad]")
            ax[0].set_title("Receding horizon: plan 0.2 s ahead, every tick", loc="left",
                            color=INK)
            ax[0].legend(loc="best")
            style(ax[0])
            im = ax[1].imshow(np.log10(np.abs(A) + 1e-12), cmap="Blues", vmin=-6)
            ax[1].set_title("log₁₀|H|: 70 × 70 Hessian (7 joints × 10 steps)", loc="left",
                            color=INK)
            ax[1].set_xticks([0, 35, 69])
            ax[1].set_yticks([0, 35, 69])
            fig.colorbar(im, ax=ax[1], shrink=0.8)
            for nm2, w2, col in (("mpc", 1e-2, C["cg"]), ("mpc_hard", 1e-3, C["plain"])):
                Hs = T.mpc_systems(w2)[ticks[1]][0]
                ev = np.sort(np.linalg.eigvalsh(Hs))[::-1]
                ax[2].semilogy(ev, "o", ms=3, color=col,
                               label=f"torque weight {w2:g} (κ ≈ {ev[0] / ev[-1]:.1e})")
            ax[2].set_xlabel("eigenvalue index")
            ax[2].set_ylabel("eigenvalue")
            ax[2].set_title("Eigenvalue spread sets the difficulty", loc="left", color=INK)
            ax[2].legend(loc="upper right")
            style(ax[2])
            fig.tight_layout()
            save(fig, "mpc_illustration.png")
        fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.4))
        for c_, i in enumerate(ticks):
            A, B, info = systems[i]
            k = info["k"]
            ref = info["xref"].reshape(H, 2 * n)[:, :n]
            Ue = sols["exact"][i]
            j = int(np.argmax(np.abs(Ue.reshape(H, n)).max(axis=0)))
            tp = dt * np.arange(1, H + 1)
            ir_lab = "thermox + IR" if b else "thermox + IR (stalls)"
            for key, col, lw, lab in (("exact", AXIS, 4, "exact solve"),
                                      ("plain", C["plain"], 1.6, f"plain thermox N={Nq:,}"),
                                      ("ir", C["ir_warm"], 1.6, ir_lab)):
                U = sols[key][i]
                X = info["Phi"] @ info["x0"] + info["Gam"] @ U
                qp = X.reshape(H, 2 * n)[:, :n]
                ax[0, c_].plot(tp, (qp[:, j] - ref[:, j]) * 1e3, "-o", color=col, lw=lw,
                               ms=3, label=lab)
                ax[1, c_].step(np.arange(H) * dt, U.reshape(H, n)[:, j], where="post",
                               color=col, lw=lw, label=lab)
            ax[0, c_].set_title(f"t = {k * E.CTRL_DT:.2f} s, joint {j + 1}", loc="left",
                                color=INK)
            ax[0, c_].set_ylabel("predicted − reference [mrad]")
            ax[1, c_].set_ylabel("planned torque [N·m]")
            ax[1, c_].set_xlabel("horizon [s]")
            for a in ax[:, c_]:
                style(a)
        ax[0, 0].legend(loc="best")
        fig.tight_layout()
        save(fig, f"solve_{name}.png")
        errs = {key: float(np.median([np.linalg.norm(sols[key][i] - sols["exact"][i]) /
                                      np.linalg.norm(sols["exact"][i]) for i in ticks]))
                for key in ("plain", "ir")}
        data[name] = dict(N_eq=Nq, cfg=b, plain_err=errs["plain"], ir_err=errs["ir"],
                          ir_rounds=float(np.mean(irs.rounds)),
                          ir_err_mean=float(np.median(irs.errs)))
        fig_convergence(name, convergence(name))
    with open(os.path.join(OUT, "data_mpc.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=float)


# ============================================================ heat equation
def part_heat():
    data = {}
    for name, n in (("heat8", 8), ("heat16", 16)):
        systems = T.PROBLEMS[name]["build"]()
        A = systems[0][0]
        d = n * n
        f_list = []
        u_prev = None
        for k, (_, b) in enumerate(systems):     # recover the source term f_k = b_k - u_k
            u_k = np.random.default_rng(0).normal(size=d) * 0.1 if k == 0 else u_prev
            f_list.append(b[:, 0] - u_k)
            u_prev = np.linalg.solve(A, b[:, 0])
        Nq = equal_budget(name)
        irs, b = ir_solver(name)
        sol = {"exact": E.ExactSolver(), "plain": plain_solver(name, Nq), "ir": irs}
        fields = {k: [] for k in sol}
        for key, s in sol.items():
            u = np.random.default_rng(0).normal(size=d) * 0.1
            for k in range(len(systems)):
                u = s(A, (u + f_list[k])[:, None])[:, 0]
                fields[key].append(u.copy())
        err = {k: [np.linalg.norm(fields[k][i] - fields["exact"][i]) /
                   np.linalg.norm(fields["exact"][i]) for i in range(len(systems))]
               for k in ("plain", "ir")}
        vmax = max(np.max(fields["exact"]), 1e-9)
        frames = []
        for i in range(len(systems)):
            fig, ax = plt.subplots(1, 4, figsize=(12.4, 3.25), dpi=90,
                                   gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
            for a, key, lab in ((ax[0], "exact", "exact solve"),
                                (ax[1], "plain", f"plain thermox N={Nq}"),
                                (ax[2], "ir", "thermox + IR")):
                a.imshow(fields[key][i].reshape(n, n).T, origin="lower", cmap="Oranges",
                         vmin=0, vmax=vmax)
                a.set_title(lab, loc="left", color=INK, fontsize=9)
                a.set_xticks([])
                a.set_yticks([])
            ax[3].semilogy(np.arange(1, i + 2), err["plain"][:i + 1], "-o", color=C["plain"],
                           ms=3, lw=1.5, label="plain thermox")
            ax[3].semilogy(np.arange(1, i + 2), np.maximum(err["ir"][:i + 1], 1e-9), "-o",
                           color=C["ir_warm"], ms=3, lw=1.5, label="thermox + IR")
            ax[3].set_xlim(0, len(systems) + 1)
            ax[3].set_ylim(1e-6, 3)
            ax[3].set_xlabel("time step")
            ax[3].set_ylabel("field error vs exact")
            ax[3].set_title(f"step {i + 1} of {len(systems)}: error vs exact", loc="left",
                            color=INK, fontsize=9)
            ax[3].legend(loc="lower right", fontsize=7.5)
            style(ax[3])
            fig.tight_layout()
            frames.append(fig_frame(fig))
            plt.close(fig)
        save_gif(frames, f"solve_{name}.gif", fps=4)
        if name == "heat8":
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
            xs = np.arange(n) / (n - 1)
            path = np.array([(0.3 + 0.4 * k / 30, 0.5 + 0.2 * np.sin(k / 5)) for k in range(30)])
            ax[0].imshow(fields["exact"][15].reshape(n, n).T, origin="lower", cmap="Oranges",
                         extent=(0, 1, 0, 1))
            ax[0].plot(path[:, 0], path[:, 1], "-", color=INK, lw=1.5)
            ax[0].plot(*path[-1], "o", color=INK, ms=6)
            ax[0].set_title("Temperature at step 16, source path", loc="left", color=INK)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].spy(A, markersize=2.5, color=C["cg"])
            ax[1].set_title("Matrix I + αL (64 × 64): 5 entries per row", loc="left",
                            color=INK)
            ax[1].set_xticks([0, 32, 63])
            ax[1].set_yticks([0, 32, 63])
            for nn, col in ((8, C["cg"]), (16, C["plain"])):
                Aa = T.heat_systems(nn)[0][0]
                ev = np.sort(np.linalg.eigvalsh(Aa))[::-1]
                ax[2].semilogy(np.linspace(0, 1, len(ev)), ev, color=col, lw=2,
                               label=f"{nn}×{nn} grid (κ ≈ {ev[0] / ev[-1]:.0f})")
            ax[2].set_xlabel("eigenvalue rank (fraction)")
            ax[2].set_ylabel("eigenvalue")
            ax[2].set_title("Eigenvalues of the step matrix", loc="left", color=INK)
            ax[2].legend(loc="upper right")
            style(ax[2])
            fig.tight_layout()
            save(fig, "heat_illustration.png")
        data[name] = dict(N_eq=Nq, cfg=b, plain_final=float(err["plain"][-1]),
                          ir_final=float(err["ir"][-1]), plain_first=float(err["plain"][0]),
                          ir_rounds=float(np.mean(irs.rounds)))
        fig_convergence(name, convergence(name))
    with open(os.path.join(OUT, "data_heat.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=float)


# ============================================================ random SPD
def part_rand(names=None):
    data = {}
    for name in names or R.RANDOM:
        systems = T.PROBLEMS[name]["build"]()
        A, B = systems[2]
        xs = np.linalg.solve(A, B)[:, 0]
        Nq = equal_budget(name)
        irs, b = ir_solver(name)
        xp = plain_solver(name, Nq)(A, B)[:, 0]
        xi = irs(A, B)[:, 0]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
        off = A - np.diag(np.diag(A))
        lim = np.percentile(np.abs(off), 99)
        ax[0].imshow(np.ma.masked_where(np.eye(len(A), dtype=bool), off), cmap="RdBu_r",
                     vmin=-lim, vmax=lim)
        ax[0].set_title(f"A off-diagonal entries ({A.shape[0]} × {A.shape[0]}, all non-zero)",
                        loc="left", color=INK, fontsize=9)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ev = np.sort(np.linalg.eigvalsh(A))[::-1]
        ax[1].semilogy(ev, "o", ms=3 if len(ev) < 100 else 1.5, color=C["cg"])
        ax[1].set_title(f"Eigenvalues: κ = {ev[0] / ev[-1]:.0f}", loc="left", color=INK)
        ax[1].set_xlabel("index")
        style(ax[1])
        lo, hi = xs.min(), xs.max()
        ax[2].plot([lo, hi], [lo, hi], color=AXIS, lw=1)
        ep = np.linalg.norm(xp - xs) / np.linalg.norm(xs)
        ei = np.linalg.norm(xi - xs) / np.linalg.norm(xs)
        ax[2].plot(xs, xp, "o", color=C["plain"], ms=3.5, alpha=0.7, mec="none",
                   label=f"plain thermox N={Nq:,}: error {100 * ep:.1f}%")
        ax[2].plot(xs, xi, "o", color=C["ir_cold"], ms=3.5, mec="none",
                   label=f"thermox + IR: error {ei:.1e}")
        ax[2].set_xlabel("exact solution entry")
        ax[2].set_ylabel("computed entry")
        ax[2].set_title("Same chip time: solution entries vs exact", loc="left", color=INK)
        ax[2].legend(loc="upper left")
        style(ax[2])
        fig.tight_layout()
        save(fig, f"solve_{name}.png")
        data[name] = dict(N_eq=Nq, cfg=b, plain_err=float(ep), ir_err=float(ei),
                          ir_rounds=irs.rounds[-1])
        fig_convergence(name, convergence(name, idx=2))
    with open(os.path.join(OUT, f"data_rand_{(names or ['all'])[0]}.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=float)


def part_results():
    import shutil
    for f in ("cost_to_accuracy", "scaling_random_spd", "wall_clock", "size_scaling",
              "nonidealities"):
        shutil.copy(os.path.join(ROOT, "results_bench", f + ".png"),
                    os.path.join(OUT, f"bench_{f}.png"))
    shutil.copy(os.path.join(ROOT, "results_ir", "pick_place_comparison.mp4"),
                os.path.join(OUT, "kf_robot.mp4"))
    shutil.copy(os.path.join(ROOT, "results_ir", "monte_carlo.png"),
                os.path.join(OUT, "kf_robot_mc.png"))
    print("copied benchmark figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--names", default=None, help="subset of random problems")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for p in a.part.split(","):
        {"method": part_method, "kf": part_kf, "ik": part_ik, "mpc": part_mpc,
         "heat": part_heat, "tables": tables, "results": part_results,
         "rand": lambda: part_rand(a.names.split(",") if a.names else None)}[p]()
