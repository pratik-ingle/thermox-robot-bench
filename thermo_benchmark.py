"""
Benchmark: thermodynamic linear solves (thermox emulation) vs. digital solvers on
robotics and classic problem sets, plus physical non-idealities of the device.

Every method solves A X = B; the reference X* is numpy's exact solve.

  chol       digital direct solve (Cholesky)                               exact
  cg / pcg   digital conjugate gradient, plain / Jacobi-preconditioned, cold or warm start
  thermo     plain thermox: one device solve of N samples from x0 = 0 (as in the robot test)
  IR         thermox device solve inside digital iterative refinement, cold or warm start

The device is thermox's own OU sampler (thermox.sampler.sample_identity_diffusion), with
the matrix pre-scaled so its top eigenvalue is 1 (one time unit = fastest device
relaxation time) and each right-hand side scaled so its largest entry uses the full
input range (every cell has its own DAC). Output voltage-swing limits are not modelled. Optional non-idealities: coupling mismatch and quantisation of the
programmed matrix, DAC bits on the input, ADC bits on every output sample.

Usage:
  python thermo_benchmark.py --problem kf_iso        # one problem set -> results_bench/
  python thermo_benchmark.py --problem all           # every problem set, sequentially
  python thermo_benchmark.py --nonideal ik:100       # non-ideality sweep (IR with N=100)
  python thermo_benchmark.py --report                # tables + figures from saved JSON
"""
import os
import json
import time
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from thermox.sampler import sample_identity_diffusion
from thermox.utils import ProcessedDriftMatrix

import kf_thermox_pick_place as E

TARGETS = [1e-2, 1e-4, 1e-6]
IR_NS = [10, 30, 100, 300, 1000]
PLAIN_NS = [10, 100, 1000, 10000]
MAX_ROUNDS = 40


# ============================================================== problem sets
class _Recorder:
    def __init__(self):
        self.systems = []

    def __call__(self, S, B):
        self.systems.append((S.copy(), B.copy()))
        return np.linalg.solve(S, B)


def _camera_R(ratio):
    """Depth-camera noise: `ratio` x larger along an oblique viewing ray."""
    v = np.array([0.6, -0.3, -0.75])
    v /= np.linalg.norm(v)
    return E.SIGMA_MEAS**2 * (np.eye(3) + (ratio**2 - 1) * np.outer(v, v))


def kf_systems(ratio, seed=0):
    """Kalman-gain systems S K^T = H P (d=3, 6 RHS) along one tracking run."""
    p_true, _ = E.truth_and_noise(seed)
    R = _camera_R(ratio)
    rng = np.random.default_rng(10_000 + seed)
    meas = p_true + rng.normal(size=p_true.shape) @ np.linalg.cholesky(R).T
    rec = _Recorder()
    kf = E.KalmanCV(rec, meas[0])
    kf.R = R
    for k in range(1, int(round(E.T_GRASP / E.CTRL_DT)) + 1):
        kf.predict(E.CTRL_DT)
        if k % E.CAM_EVERY == 0:
            kf.update(meas[k])
    return rec.systems


_ROBOT_LOG = None


def robot_log():
    """One exact-solver pick-and-place run: IK systems + joint states / mass matrices."""
    global _ROBOT_LOG
    if _ROBOT_LOG is not None:
        return _ROBOT_LOG
    model = E.build_model()
    ik_sys = []

    class RecIK(E.IK):
        def solve(self, q, p_des, R_des, iters=4, lam=0.05):
            d = self.d
            d.qpos[:7] = q
            for _ in range(iters):
                mujoco.mj_kinematics(self.m, d)
                mujoco.mj_comPos(self.m, d)
                R = d.site_xmat[self.sid].reshape(3, 3)
                ep = p_des - d.site_xpos[self.sid]
                Re = R_des @ R.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0],
                                     Re[1, 0] - Re[0, 1]])
                mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.sid)
                J = np.vstack([self.jp[:, :7], self.jr[:, :7]])
                A = J @ J.T + lam**2 * np.eye(6)
                e = np.r_[ep, er]
                ik_sys.append((A, e[:, None].copy()))
                dq = J.T @ np.linalg.solve(A, e)
                d.qpos[:7] = np.clip(d.qpos[:7] + dq, self.lo, self.hi)
            return d.qpos[:7].copy()

    log = {k: [] for k in ("q", "qd", "qdes", "M")}
    Mbuf = np.zeros((model.nv, model.nv))

    def cb(k, t, m, d, L):
        if hasattr(d, "qM"):
            mujoco.mj_fullM(m, Mbuf, d.qM)
        else:                                  # MuJoCo >= 3.5 API
            mujoco.mj_fullM(m, d, Mbuf)
        log["q"].append(d.qpos[:7].copy())
        log["qd"].append(d.qvel[:7].copy())
        log["qdes"].append(d.ctrl[:7].copy())
        log["M"].append(Mbuf[:7, :7].copy())

    orig, E.IK = E.IK, RecIK
    try:
        E.run_trial(model, E.ExactSolver(), 0, frame_cb=cb)
    finally:
        E.IK = orig
    _ROBOT_LOG = ik_sys, {k: np.array(v) for k, v in log.items()}
    return _ROBOT_LOG


def ik_systems():
    """Damped-least-squares IK systems (J J^T + lam^2 I) y = e (d=6), approach phase.
    Only the first of the 4 IK iterations per tick: the later ones are micrometre-sized
    corrections (or exactly zero), where a relative-error target is meaningless."""
    ik_sys, _ = robot_log()
    k0, k1 = int(E.T_TRACK / E.CTRL_DT), int(E.T_GRASP / E.CTRL_DT)
    return [s for s in ik_sys[4 * k0: 4 * k1: 4] if np.linalg.norm(s[1]) > 0]


MPC_H, MPC_STRIDE = 10, 2                  # 10 steps x 0.02 s horizon
MPC_WQ, MPC_WV = 1e3, 1.0


def mpc_systems(MPC_WR, every=6, parts=False):
    """Condensed joint-torque MPC Hessian systems H U = -g (d = 7 x 10 = 70) along the run,
    linearised with the Panda mass matrix M(q) at each tick. MPC_WR (torque weight)
    sets the conditioning. parts=True also returns the prediction matrices per system."""
    _, log = robot_log()
    n, H, dt = 7, MPC_H, MPC_STRIDE * E.CTRL_DT
    Ad = np.block([[np.eye(n), dt * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
    Qx = np.diag([MPC_WQ] * n + [MPC_WV] * n)
    systems = []
    k_end = len(log["q"]) - H * MPC_STRIDE
    for k in range(int(E.T_TRACK / E.CTRL_DT), k_end, every):
        Minv = np.linalg.inv(log["M"][k])
        Bd = np.vstack([0.5 * dt**2 * Minv, dt * Minv])
        Phi = np.zeros((2 * n * H, 2 * n))
        Gam = np.zeros((2 * n * H, n * H))
        Ap = np.eye(2 * n)
        for t in range(H):
            Ap = Ad @ Ap
            Phi[2 * n * t: 2 * n * (t + 1)] = Ap
            for j in range(t + 1):
                Gam[2 * n * t: 2 * n * (t + 1), n * j: n * (j + 1)] = \
                    np.linalg.matrix_power(Ad, t - j) @ Bd
        Qbar = np.kron(np.eye(H), Qx)
        x0 = np.r_[log["q"][k], log["qd"][k]]
        xref = np.concatenate([np.r_[log["qdes"][k + MPC_STRIDE * (t + 1)], np.zeros(n)]
                               for t in range(H)])
        Hm = Gam.T @ Qbar @ Gam + MPC_WR * np.eye(n * H)
        g = Gam.T @ Qbar @ (Phi @ x0 - xref)
        system = (0.5 * (Hm + Hm.T), -g[:, None])
        if parts:
            system += (dict(k=k, Phi=Phi, Gam=Gam, x0=x0, xref=xref),)
        systems.append(system)
    return systems


def heat_systems(n, alpha=12.0, steps=30, seed=0):
    """Implicit-Euler heat equation on an n x n grid with a moving source:
    (I + alpha L) u_{k+1} = u_k + f_k  (constant matrix, d = n^2)."""
    d = n * n
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    L = np.kron(T, np.eye(n)) + np.kron(np.eye(n), T)
    A = np.eye(d) + alpha * L
    xs = np.arange(n) / (n - 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = np.random.default_rng(seed).normal(size=d) * 0.1
    systems = []
    for k in range(steps):
        cx, cy = 0.3 + 0.4 * k / steps, 0.5 + 0.2 * np.sin(k / 5)
        f = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.02).ravel()
        b = u + f
        systems.append((A, b[:, None]))
        u = np.linalg.solve(A, b)
    return systems


def random_spd_systems(d, kappa, count=5, seed=0):
    """Independent dense SPD systems with log-spaced eigenvalues in [1, kappa]."""
    rng = np.random.default_rng(seed + 1000 * d + int(kappa))
    out = []
    for _ in range(count):
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        lam = np.logspace(0, np.log10(kappa), d)
        A = (Q * lam) @ Q.T
        out.append((0.5 * (A + A.T), rng.normal(size=(d, 1))))
    return out


PROBLEMS = {
    "kf_iso":     dict(build=lambda: kf_systems(1.0), seq=True,
                       desc="KF gain, isotropic camera (d=3, 6 RHS)"),
    "kf_depth3":  dict(build=lambda: kf_systems(3.0), seq=True,
                       desc="KF gain, depth noise 3x (d=3, 6 RHS)"),
    "kf_depth10": dict(build=lambda: kf_systems(10.0), seq=True,
                       desc="KF gain, depth noise 10x (d=3, 6 RHS)"),
    "ik":         dict(build=ik_systems, seq=True,
                       desc="Arm IK, damped least squares (d=6)"),
    "mpc":        dict(build=lambda: mpc_systems(1e-2), seq=True,
                       desc="Joint-torque MPC Hessian (d=70)"),
    "mpc_hard":   dict(build=lambda: mpc_systems(1e-3), seq=True,
                       desc="Joint-torque MPC Hessian, low torque weight (d=70)"),
    "heat8":      dict(build=lambda: heat_systems(8), seq=True,
                       desc="Heat equation 8x8 grid (d=64)"),
    "heat16":     dict(build=lambda: heat_systems(16), seq=True,
                       desc="Heat equation 16x16 grid (d=256)"),
}
for _d in (16, 64, 256):
    for _k in (10, 100, 1000):
        PROBLEMS[f"rand_d{_d}_k{_k}"] = dict(
            build=partial(random_spd_systems, _d, _k), seq=False,
            desc=f"Random SPD d={_d}, cond={_k}")


# ============================================================== digital solvers
def rel_err(X, Xs):
    return float(np.linalg.norm(X - Xs) / np.linalg.norm(Xs))


def run_cg(systems, target, warm, jacobi, maxit=None):
    """Block CG (columns in lock-step), stopped when the true relative error <= target."""
    out, X_prev = [], None
    for A, B in systems:
        Xs = np.linalg.solve(A, B)
        X = X_prev.copy() if warm and X_prev is not None else np.zeros_like(B)
        Minv = (1.0 / np.diag(A))[:, None] if jacobi else np.ones((A.shape[0], 1))
        R = B - A @ X
        Z = Minv * R
        P = Z.copy()
        rz = np.sum(R * Z, axis=0)
        it, lim = 0, maxit or 20 * A.shape[0]
        while rel_err(X, Xs) > target and it < lim:
            AP = A @ P
            pap = np.sum(P * AP, axis=0)
            alpha = np.divide(rz, pap, out=np.zeros_like(rz), where=pap > 0)
            X = X + alpha * P
            R = R - alpha * AP
            Z = Minv * R
            rz_new = np.sum(R * Z, axis=0)
            beta = np.divide(rz_new, rz, out=np.zeros_like(rz), where=rz > 0)
            P = Z + beta * P
            rz = rz_new
            it += 1
        out.append(dict(iters=it, err=rel_err(X, Xs)))
        X_prev = X
    return out


# ============================================================== thermodynamic device
@partial(jax.jit, static_argnums=(4, 5))
def _device_avg(proc, mean, key, out_q, N, quantise_out):
    """Time-average of N+1 OU samples per column (x0 = 0 included, exactly as
    thermox.linalg.solve does). out_q = (full_scale, lsb) for the per-sample ADC."""
    ts = jnp.concatenate([jnp.zeros(1), jnp.arange(N + 1, dtype=float)])
    keys = jax.random.split(key, mean.shape[1])

    def one(m, k):
        xs = sample_identity_diffusion(k, ts, jnp.zeros_like(m), proc, m)[1:]
        if quantise_out:
            fs, lsb = out_q
            xs = jnp.clip(jnp.round(xs / lsb) * lsb, -fs, fs)
        return jnp.mean(xs, axis=0)

    return jax.vmap(one, in_axes=(1, 0), out_axes=1)(mean, keys)


def _quantise(x, bits):
    lsb = np.max(np.abs(x)) / 2 ** (bits - 1)
    return x if lsb == 0 else np.round(x / lsb) * lsb


class Device:
    """Emulated thermodynamic chip: programmed with a matrix (eig <= 1), then asked for
    time-averaged solutions. Mismatch is a fixed relative error per coupling element."""

    def __init__(self, N, seed=0, mismatch=0.0, bits=None):
        self.N, self.mismatch, self.bits = N, mismatch, bits
        self.key = jax.random.PRNGKey(seed)
        self.mm_rng = np.random.default_rng(777 + seed)
        self._mm = None

    def program(self, A):
        d = A.shape[0]
        if self.mismatch:
            if self._mm is None or self._mm.shape[0] != d:
                G = self.mm_rng.normal(size=(d, d))
                self._mm = (G + G.T) / np.sqrt(2)
            A = A * (1 + self.mismatch * self._mm)
        if self.bits:
            A = _quantise(A, self.bits)
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        if w[0] <= 0:
            self.proc = None              # programmed matrix not positive definite:
            return False                  # the physical OU process would diverge
        self.proc = ProcessedDriftMatrix(jnp.asarray(A), jnp.asarray(w), jnp.asarray(V),
                                         jnp.asarray(V.T), jnp.asarray(w), jnp.asarray(V))
        self.Ainv = (V / w) @ V.T
        self.sd_max = float(np.sqrt(np.max(np.diag(self.Ainv)) / 2))
        return True

    def solve(self, Bn):
        if self.bits:
            Bn = _quantise(Bn, self.bits)
        mean = self.Ainv @ Bn
        fs = float(np.max(np.abs(mean)) + 4 * self.sd_max)
        q = (fs, 2 * fs / 2 ** self.bits) if self.bits else (1.0, 1.0)
        self.key, sub = jax.random.split(self.key)
        return np.asarray(_device_avg(self.proc, jnp.asarray(mean), sub, q, self.N,
                                      bool(self.bits)))


def run_thermo(systems, N, target, warm, jacobi=True, rounds_max=MAX_ROUNDS, seed=0,
               **device_kw):
    """Device solve inside digital iterative refinement (rounds_max=1, target=0 -> plain).
    Rounds stop once the true relative error <= target (oracle stopping, as for CG)."""
    dev = Device(N, seed, **device_kw)
    out, X_prev, A_prev = [], None, None
    for A, B in systems:
        Xs = np.linalg.solve(A, B)
        new = A is not A_prev
        if new:
            P = 1 / np.sqrt(np.diag(A)) if jacobi else np.ones(A.shape[0])
            As = P[:, None] * A * P[None, :]
            # top eigenvalue -> 1 (device bandwidth). On hardware this is a few warm-started
            # power iterations; the cheaper Gershgorin bound overestimates it ~10x for dense A.
            c = np.linalg.eigvalsh(As)[-1]
            stable = dev.program(As / c)
            A_prev = A
        X = X_prev.copy() if warm and X_prev is not None else np.zeros_like(B)
        errs = [rel_err(X, Xs)]
        failed = not stable
        while not failed and len(errs) <= rounds_max and errs[-1] > target:
            Rr = P[:, None] * (B - A @ X) / c
            s = np.max(np.abs(Rr), axis=0)        # largest input at each DAC's full range
            s = np.where(s > 0, s, 1.0)
            X = X + P[:, None] * (dev.solve(Rr / s) * s)
            errs.append(rel_err(X, Xs))
            if not np.isfinite(errs[-1]) or errs[-1] > 1e3 * max(errs[0], 1e-12):
                failed = True
            elif len(errs) > 6 and errs[-1] > 0.5 * errs[-6]:
                break                 # stalled: < 2x gain over 5 rounds (device eps ~ 1)
        out.append(dict(rounds=len(errs) - 1, err=errs[-1], new_matrix=bool(new),
                        failed=bool(failed)))
        X_prev = X if np.all(np.isfinite(X)) else None
    return out


# ============================================================== experiments
def problem_stats(systems):
    conds = [np.linalg.cond(A) for A, _ in systems]
    Ps = [1 / np.sqrt(np.diag(A)) for A, _ in systems]
    pconds = [np.linalg.cond(p[:, None] * A * p[None, :]) for p, (A, _) in zip(Ps, systems)]
    A0, B0 = systems[0]
    return dict(n_systems=len(systems), d=A0.shape[0], m=B0.shape[1],
                cond_median=float(np.median(conds)), cond_max=float(np.max(conds)),
                jacobi_cond_median=float(np.median(pconds)),
                n_matrices=len({id(A) for A, _ in systems}))


def summarise(rows, key):
    v = np.array([r[key] for r in rows], float)
    return float(np.mean(v))


def run_problem(name, out_dir):
    spec = PROBLEMS[name]
    systems = spec["build"]()
    res = dict(name=name, desc=spec["desc"], seq=spec["seq"], **problem_stats(systems))
    print(f"[{name}] {res['desc']}: {res['n_systems']} systems, "
          f"cond median {res['cond_median']:.3g} (Jacobi {res['jacobi_cond_median']:.3g})",
          flush=True)
    warms = [False, True] if spec["seq"] else [False]
    t0 = time.time()

    res["plain"] = {}
    for N in PLAIN_NS:
        for jac in (False, True):
            rows = run_thermo(systems, N, target=0.0, warm=False, jacobi=jac, rounds_max=1)
            res["plain"][f"{N}|{'jac' if jac else 'none'}"] = dict(
                err=summarise(rows, "err"), failed=summarise(rows, "failed"))
        print(f"  plain N={N:<6d} err " + " / ".join(
            f"{res['plain'][f'{N}|{p}']['err']:.3g} ({p})" for p in ("none", "jac")), flush=True)

    res["ir"], res["cg"] = {}, {}
    for tgt in TARGETS:
        for warm in warms:
            w = "warm" if warm else "cold"
            for jac in (False, True):
                for N in IR_NS:
                    rows = run_thermo(systems, N, tgt, warm, jacobi=jac)
                    res["ir"][f"{tgt:g}|{w}|{N}|{'jac' if jac else 'none'}"] = dict(
                        rounds=summarise(rows, "rounds"),
                        rounds_max=float(np.max([r["rounds"] for r in rows])),
                        reached=float(np.mean([r["err"] <= tgt for r in rows])),
                        failed=summarise(rows, "failed"),
                        new_matrix=summarise(rows, "new_matrix"))
                rows = run_cg(systems, tgt, warm, jac)
                res["cg"][f"{tgt:g}|{w}|{'pcg' if jac else 'cg'}"] = dict(
                    iters=summarise(rows, "iters"),
                    reached=float(np.mean([r["err"] <= tgt for r in rows])))
            for p in ("none", "jac"):
                print(f"  target {tgt:g} {w} {p:4s}: IR rounds (reached) " + ", ".join(
                    f"N={N}:{res['ir'][f'{tgt:g}|{w}|{N}|{p}']['rounds']:.1f}"
                    f"({100 * res['ir'][f'{tgt:g}|{w}|{N}|{p}']['reached']:.0f}%)"
                    for N in IR_NS), flush=True)
            print(f"  target {tgt:g} {w}: CG iters {res['cg'][f'{tgt:g}|{w}|cg']['iters']:.1f}"
                  f" | PCG iters {res['cg'][f'{tgt:g}|{w}|pcg']['iters']:.1f}", flush=True)
    res["runtime_s"] = time.time() - t0
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{name}] done in {res['runtime_s']:.0f}s", flush=True)


NONIDEAL = dict(mismatch=[0.0, 0.01, 0.03, 0.1], bits=[6, 8, 10, 12])


def run_nonideal(name, out_dir, max_systems=60, N_plain=1000, N_ir=30, target=1e-4):
    spec = PROBLEMS[name]
    systems = spec["build"]()[:max_systems]
    warm = spec["seq"]
    res = dict(name=name, desc=spec["desc"], N_plain=N_plain, N_ir=N_ir, target=target,
               warm=warm, **problem_stats(systems), sweeps={})
    for kind, values in NONIDEAL.items():
        for v in values:
            kw = {kind: v}
            plain = run_thermo(systems, N_plain, 0.0, False, rounds_max=1, **kw)
            ir = run_thermo(systems, N_ir, target, warm, **kw)
            row = dict(plain_err=float(np.median([r["err"] for r in plain])),
                       ir_rounds=summarise(ir, "rounds"),
                       ir_reached=float(np.mean([r["err"] <= target for r in ir])),
                       ir_err_median=float(np.median([r["err"] for r in ir])),
                       unstable=summarise(ir, "failed"))
            res["sweeps"][f"{kind}={v}"] = row
            print(f"[{name}] {kind}={v}: plain err {row['plain_err']:.3g} | IR rounds "
                  f"{row['ir_rounds']:.2f}, reached {100 * row['ir_reached']:.0f}%, "
                  f"failed {100 * row['unstable']:.0f}%", flush=True)
    with open(os.path.join(out_dir, f"nonideal_{name}.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", help="problem set name or 'all'")
    ap.add_argument("--nonideal", help="problem set for the non-ideality sweep")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="results_bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.problem:
        for name in (PROBLEMS if a.problem == "all" else a.problem.split(",")):
            run_problem(name, a.out)
    if a.nonideal:
        for spec in a.nonideal.split(","):                # name or name:N_ir
            name, _, n_ir = spec.partition(":")
            run_nonideal(name, a.out, N_ir=int(n_ir) if n_ir else 30)
    if a.report:
        import thermo_benchmark_report
        thermo_benchmark_report.report(a.out)


if __name__ == "__main__":
    main()
