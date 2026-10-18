"""
Conveyor pick-and-place with a Kalman-filter object tracker.

A Franka Panda (MuJoCo, physics-stepped, joint position servos) must pick a cube
moving on a conveyor. A noisy "camera" measures the cube position at ~33 Hz; a
constant-velocity Kalman filter estimates position + velocity, and the planner
continuously re-predicts the intercept point and drives the arm there.

The only thing that differs between methods is how the Kalman gain is computed:

    K^T = S^{-1} (H P),   S = H P H^T + R      (3x3 SPD system, 6 right-hand sides)

  * exact  : numpy.linalg.solve
  * thermo : thermox.linalg.solve  (time-average of an Ornstein-Uhlenbeck process,
             i.e. an emulation of a thermodynamic linear-algebra chip), with
             num_samples = N as the "hardware time" budget.
  * thermo IR : the same device solve wrapped in digital iterative refinement,
             optionally warm-started from the previous gain (budget = steps * N).

All methods share identical cube motion and camera noise per seed (common random
numbers), so differences come only from the solver.

Usage:
    python kf_thermox_pick_place.py --mc 25      # Monte Carlo + figures + video
    python kf_thermox_pick_place.py --ir 3x10w,3x33w   # refinement configs (steps x N [w=warm])
"""
import os
import time
import json
import argparse

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import jax
import jax.numpy as jnp  # noqa: F401
import thermox
import mujoco

jax.config.update("jax_enable_x64", True)

PANDA_DIR = os.environ.get("PANDA_DIR", "menag/franka_emika_panda")

# ----------------------------------------------------------------- parameters
SIM_DT = 0.002
CTRL_DT = 0.01            # control + KF predict at 100 Hz
CAM_EVERY = 3             # camera update every 3rd tick (~33 Hz)
SIGMA_MEAS = 0.015        # camera noise std [m]
Q_TRUE = 2e-4           # true velocity random-walk intensity [m^2/s^3] (~1.4 cm/s drift)
Q_KF = 2e-4             # KF white-noise-acceleration PSD (matched)
BELT_SPEED = 0.08         # [m/s]
CUBE_HALF = 0.02
BELT_X, BELT_Z = 0.55, 0.10
CUBE_START = np.array([BELT_X, -0.45, BELT_Z + CUBE_HALF])
BIN_XY = np.array([0.30, 0.45])
BIN_FLOOR = 0.01
HOVER = 0.12

T_TRACK = 1.5             # watch only
T_APPROACH = 2.0          # intercept motion
T_GRASP = T_TRACK + T_APPROACH
T_CLOSE, T_LIFT, T_MOVE, T_LOWER, T_RETREAT = 0.25, 0.6, 1.6, 0.6, 0.8
T_END = T_GRASP + T_CLOSE + T_LIFT + T_MOVE + T_LOWER + T_RETREAT
GRASP_TOL = 0.015         # pick succeeds if TCP within 1.5 cm of true cube centre

N_TICKS = int(round(T_END / CTRL_DT))


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ------------------------------------------------------------------ solvers
class ExactSolver:
    label = "Exact (numpy)"
    key = "exact"

    def __call__(self, S, B):
        return np.linalg.solve(S, B)


_THERMO_CACHE = {}


def _thermo_fn(N):
    if N not in _THERMO_CACHE:
        def _solve(A, B, key):
            keys = jax.random.split(key, B.shape[1])
            return jax.vmap(
                lambda b, k: thermox.linalg.solve(A, b, num_samples=N, key=k),
                in_axes=(1, 0), out_axes=1)(B, keys)
        _THERMO_CACHE[N] = jax.jit(_solve)
    return _THERMO_CACHE[N]


class ThermoSolver:
    def __init__(self, num_samples, seed=0):
        self.N = num_samples
        self.label = f"thermox N={num_samples}"
        self.key = f"thermo_{num_samples}"
        self._fn = _thermo_fn(num_samples)
        self._rng = jax.random.PRNGKey(100_000 + seed)

    def _device_solve(self, S, B):
        # Map the problem onto the "device" dynamic range (as an analog chip would):
        #  - scale A so eig(A) ~ 1 (OU relaxation time ~ 1 time unit)
        #  - scale each RHS column to unit norm, so the sampling noise is relative
        c = np.trace(S) / S.shape[0]
        n = np.linalg.norm(B, axis=0)
        s = np.where(n > 0, n, 1.0) / c
        self._rng, sub = jax.random.split(self._rng)
        return np.asarray(self._fn(S / c, B / (c * s), sub)) * s

    def __call__(self, S, B):
        return self._device_solve(S, B)


class ThermoRefineSolver(ThermoSolver):
    """thermox solve inside digital mixed-precision iterative refinement.

    Each step the device solves for the correction, S d = B - S X, with the residual
    re-normalised to the device's full range, so the error shrinks by the solver's
    relative error eps(N) per step (eps^steps) instead of 1/sqrt(N). With warm=True
    the first iterate is the previous gain: the KF gain changes little between camera
    frames, so the device only has to resolve that small change. The residual is a
    cheap O(d^2) digital matvec; the O(d^3) solve stays on the device.
    Total device time per gain = steps * N.
    """

    def __init__(self, num_samples, steps=3, warm=True, seed=0):
        super().__init__(num_samples, seed)
        self.steps, self.warm = steps, warm
        self.label = f"thermox IR {steps}×{num_samples}" + (" warm" if warm else "")
        self.key = ir_key(num_samples, steps, warm)
        self._X = None

    def reset(self):
        self._X = None

    def __call__(self, S, B):
        X = self._X if self.warm and self._X is not None else np.zeros_like(B)
        for _ in range(self.steps):
            X = X + self._device_solve(S, B - S @ X)
        self._X = X
        return X


def ir_key(N, steps, warm):
    return f"thermo_ir{steps}x{N}" + ("w" if warm else "")


def parse_ir(spec):
    """'3x10w' -> (N=10, steps=3, warm=True)"""
    steps, rest = spec.split("x")
    return int(rest.rstrip("w")), int(steps), rest.endswith("w")


# ------------------------------------------------------------- Kalman filter
class KalmanCV:
    """Constant-velocity KF, state [p, v] in R^6, measures p."""

    def __init__(self, solver, z0):
        self.solver = solver
        self.x = np.r_[z0, 0.0, 0.0, 0.0]
        self.P = np.diag([SIGMA_MEAS**2] * 3 + [0.1**2] * 3)
        self.R = SIGMA_MEAS**2 * np.eye(3)
        self.gain_err, self.solve_s = [], []

    def predict(self, dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        I3 = np.eye(3)
        Q = Q_KF * np.block([[dt**3 / 3 * I3, dt**2 / 2 * I3],
                             [dt**2 / 2 * I3, dt * I3]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        S = self.P[:3, :3] + self.R          # H P H^T + R
        HP = self.P[:3, :]                   # H P
        t0 = time.perf_counter()
        X = self.solver(S, HP)               # S^{-1} H P
        self.solve_s.append(time.perf_counter() - t0)
        X_ref = np.linalg.solve(S, HP)       # diagnostics only
        self.gain_err.append(np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref))
        K = X.T
        self.x = self.x + K @ (z - self.x[:3])
        IKH = np.eye(6)
        IKH[:, :3] -= K
        # Joseph form keeps P SPD even when K is suboptimal (noisy solver)
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ------------------------------------------------------------------- scene
def build_model():
    spec = mujoco.MjSpec.from_file(os.path.join(PANDA_DIR, "scene.xml"))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034])
    wb = spec.worldbody
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    vis = dict(contype=0, conaffinity=0)
    wb.add_geom(type=BOX, size=[0.09, 0.65, BELT_Z / 2], pos=[BELT_X, 0, BELT_Z / 2],
                rgba=[0.12, 0.12, 0.13, 1], **vis)
    for dx in (-0.095, 0.095):
        wb.add_geom(type=BOX, size=[0.006, 0.65, 0.012], pos=[BELT_X + dx, 0, BELT_Z + 0.006],
                    rgba=[0.75, 0.75, 0.78, 1], **vis)
    bx, by = BIN_XY
    wb.add_geom(type=BOX, size=[0.09, 0.09, BIN_FLOOR / 2], pos=[bx, by, BIN_FLOOR / 2],
                rgba=[0.2, 0.45, 0.8, 1], **vis)
    for (ox, oy, sx, sy) in [(0.09, 0, 0.005, 0.09), (-0.09, 0, 0.005, 0.09),
                             (0, 0.09, 0.09, 0.005), (0, -0.09, 0.09, 0.005)]:
        wb.add_geom(type=BOX, size=[sx, sy, 0.04], pos=[bx + ox, by + oy, 0.04],
                    rgba=[0.2, 0.45, 0.8, 1], **vis)
    cube = wb.add_body(name="cube", mocap=True, pos=CUBE_START.tolist())
    cube.add_geom(type=BOX, size=[CUBE_HALF] * 3, rgba=[0.92, 0.25, 0.2, 1], **vis)
    return spec.compile()


class IK:
    """Damped-least-squares IK on TCP position + orientation."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.lo = model.jnt_range[:7, 0]
        self.hi = model.jnt_range[:7, 1]
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def solve(self, q, p_des, R_des, iters=4, lam=0.05):
        d = self.d
        d.qpos[:7] = q
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            R = d.site_xmat[self.sid].reshape(3, 3)
            ep = p_des - d.site_xpos[self.sid]
            Re = R_des @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, :7], self.jr[:, :7]])
            e = np.r_[ep, er]
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, self.lo, self.hi)
        return d.qpos[:7].copy()


# ----------------------------------------------------------------- one trial
def truth_and_noise(seed):
    """Cube path on belt + camera noise, identical for every method (CRN)."""
    rng = np.random.default_rng(seed)
    p = np.zeros((N_TICKS + 1, 3))
    v = np.array([0.0, BELT_SPEED])
    p[0] = CUBE_START
    for k in range(1, N_TICKS + 1):
        v = v - 0.5 * (v - [0.0, BELT_SPEED]) * CTRL_DT \
            + np.sqrt(Q_TRUE * CTRL_DT) * rng.normal(size=2)
        p[k] = p[k - 1] + np.r_[v, 0.0] * CTRL_DT
    meas = p + SIGMA_MEAS * rng.normal(size=p.shape)
    return p, meas


def run_trial(model, solver, seed, frame_cb=None):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ik = IK(model)
    sid = ik.sid
    n_sub = int(round(CTRL_DT / SIM_DT))
    tcp_home = data.site_xpos[sid].copy()
    R_des = data.site_xmat[sid].reshape(3, 3).copy()
    q_des = data.qpos[:7].copy()

    p_true, meas = truth_and_noise(seed)
    kf = KalmanCV(solver, meas[0])

    L = {k: np.full((N_TICKS + 1, 3), np.nan) for k in
         ("p_true", "x_hat", "v_hat", "tcp", "tcp_cmd", "cube", "meas", "intercept")}
    attached, grasp_err, attach_off = False, None, None
    p_grasp = None
    cube_drop = None
    cube_vz = 0.0
    start_appr = tcp_home.copy()

    t_lift0 = T_GRASP + T_CLOSE
    t_move0 = t_lift0 + T_LIFT
    t_low0 = t_move0 + T_MOVE
    t_ret0 = t_low0 + T_LOWER

    for k in range(N_TICKS + 1):
        t = k * CTRL_DT
        tcp = data.site_xpos[sid].copy()

        # ---------------- perception + estimation (until grasp)
        if t <= T_GRASP + 1e-9:
            if k > 0:
                kf.predict(CTRL_DT)
            if k % CAM_EVERY == 0 and k > 0:
                kf.update(meas[k])
                L["meas"][k] = meas[k]
            L["x_hat"][k] = kf.x[:3]
            L["v_hat"][k] = kf.x[3:]

        # ---------------- grasp decision
        if p_grasp is None and t >= T_GRASP - 1e-9:
            grasp_err = float(np.linalg.norm(tcp - p_true[k]))
            p_grasp = tcp.copy()
            if grasp_err < GRASP_TOL:
                attached = True
                attach_off = p_true[k] - tcp

        # ---------------- Cartesian plan
        if t < T_TRACK:
            cmd = tcp_home
        elif t <= T_GRASP:
            tau = (t - T_TRACK) / T_APPROACH
            g = kf.x[:3] + kf.x[3:] * (T_GRASP - t)          # predicted intercept
            L["intercept"][k] = g
            h = HOVER * (1 - minjerk((tau - 0.5) / 0.5))
            cmd = start_appr + minjerk(tau) * (g + [0, 0, h] - start_appr)
        else:
            lift = p_grasp + [0, 0, 0.20]
            above_bin = np.r_[BIN_XY, lift[2]]
            in_bin = np.r_[BIN_XY, BIN_FLOOR + CUBE_HALF + 0.04]
            if t < t_lift0:
                cmd = p_grasp
            elif t < t_move0:
                cmd = p_grasp + minjerk((t - t_lift0) / T_LIFT) * (lift - p_grasp)
            elif t < t_low0:
                cmd = lift + minjerk((t - t_move0) / T_MOVE) * (above_bin - lift)
            elif t < t_ret0:
                cmd = above_bin + minjerk((t - t_low0) / T_LOWER) * (in_bin - above_bin)
            else:
                cmd = in_bin + minjerk((t - t_ret0) / T_RETREAT) * (above_bin - in_bin)

        q_des = ik.solve(q_des, cmd, R_des)
        data.ctrl[:7] = q_des                     # Panda joint position servos
        closed = T_GRASP <= t < t_ret0
        data.ctrl[7] = 127 if closed else 255

        # ---------------- cube (kinematic): belt -> gripper -> bin
        if attached and t < t_ret0:
            cube = tcp + attach_off
        elif attached:
            if cube_drop is None:
                cube_drop = tcp + attach_off
            cube_vz -= 9.81 * CTRL_DT
            cube_drop[2] = max(BIN_FLOOR + CUBE_HALF, cube_drop[2] + cube_vz * CTRL_DT)
            cube = cube_drop
        else:
            cube = p_true[k]
        data.mocap_pos[0] = cube

        L["p_true"][k], L["tcp"][k], L["tcp_cmd"][k], L["cube"][k] = p_true[k], tcp, cmd, cube

        if frame_cb is not None:
            frame_cb(k, t, model, data, L)

        for _ in range(n_sub):
            mujoco.mj_step(model, data)

    est = slice(int(0.5 / CTRL_DT), int(T_GRASP / CTRL_DT) + 1)
    pos_err = np.linalg.norm(L["x_hat"][est] - L["p_true"][est], axis=1)
    true_v = np.gradient(p_true, CTRL_DT, axis=0)
    vel_err = np.linalg.norm(L["v_hat"][est] - true_v[est], axis=1)
    metrics = dict(
        method=solver.key, seed=seed,
        grasp_err=grasp_err, success=bool(grasp_err < GRASP_TOL),
        pos_rmse=float(np.sqrt(np.mean(pos_err**2))),
        vel_rmse=float(np.sqrt(np.mean(vel_err**2))),
        gain_err=float(np.mean(kf.gain_err)),
        solve_ms=float(1e3 * np.median(kf.solve_s)),
        n_updates=len(kf.solve_s))
    L["gain_err"] = np.array(kf.gain_err)
    return metrics, L


# ----------------------------------------------------------------- rendering
COLORS = {"exact": (0.20, 0.55, 1.00), "thermo_10000": (1.00, 0.60, 0.10),
          "thermo_10": (0.95, 0.20, 0.45), "thermo_ir3x10w": (0.70, 0.40, 1.00)}


def _line(scene, a, b, rgba, w):
    if scene.ngeom >= scene.maxgeom or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, a, b)
    scene.ngeom += 1


def _ball(scene, p, rgba, r):
    if scene.ngeom >= scene.maxgeom or np.any(np.isnan(p)):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p,
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def make_recorder(renderer, color, every=8):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.42, 0.05, 0.26]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 160, -38
    frames = []

    def cb(k, t, m, d, L):
        if k % every:
            return
        renderer.update_scene(d, camera=cam)
        sc = renderer.scene
        ks = range(0, k, 3)
        for a, b in zip(list(ks)[:-1], list(ks)[1:]):
            _line(sc, L["p_true"][a], L["p_true"][b], (1, 1, 1, 1), 0.004)
            _line(sc, L["x_hat"][a], L["x_hat"][b], (*color, 1), 0.006)
            _line(sc, L["tcp"][a], L["tcp"][b], (0.55, 1.0, 0.2, 1), 0.0045)
        for j in range(max(0, k - 45), k + 1):
            _ball(sc, L["meas"][j], (1, 0.1, 0.1, 1), 0.007)
        _ball(sc, L["intercept"][k], (1, 0.9, 0.1, 1), 0.012)
        frames.append(renderer.render().copy())

    return cb, frames


def compose_video(panels, path, fps=15):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    n = min(len(p["frames"]) for p in panels)
    out = []
    for i in range(n):
        tiles = []
        for p in panels:
            im = Image.fromarray(p["frames"][i])
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, im.width, 46], fill=(0, 0, 0))
            dr.text((8, 4), p["title"], fill=tuple(int(255 * c) for c in p["color"]), font=font)
            t = i * p["every"] * CTRL_DT
            status = f"t = {t:4.2f} s"
            if t >= T_GRASP:
                status += f"   grasp err {1e3 * p['grasp_err']:.1f} mm  " + \
                          ("PICKED" if p["grasp_err"] < GRASP_TOL else "MISSED")
            dr.text((8, 26), status, fill=(230, 230, 230), font=small)
            tiles.append(np.asarray(im))
        out.append(np.concatenate(tiles, axis=1))
    leg = Image.new("RGB", (out[0].shape[1], 24), (0, 0, 0))
    dr = ImageDraw.Draw(leg)
    dr.text((8, 5), "white: true cube path   colour: KF estimate   green: gripper (TCP) path   "
                    "red dots: camera measurements   yellow: predicted intercept",
            fill=(220, 220, 220), font=small)
    out = [np.concatenate([f, np.asarray(leg)], axis=0) for f in out]
    pil = [Image.fromarray(f).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in out]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps),
                loop=0, optimize=True)
    try:
        imageio.mimsave(path.replace(".gif", ".mp4"), out, fps=fps, quality=8,
                        macro_block_size=8)
    except Exception as e:  # imageio-ffmpeg not installed
        print("mp4 skipped:", e)


# ----------------------------------------------------------------- figures
def plot_showcase(logs, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 9.5))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axz = fig.add_subplot(2, 2, 2)
    axe = fig.add_subplot(2, 2, 3)
    axg = fig.add_subplot(2, 2, 4)
    t = np.arange(N_TICKS + 1) * CTRL_DT
    first = next(iter(logs.values()))
    ax3.plot(*first["p_true"].T, "k-", lw=1.5, label="true cube (belt)")
    m = ~np.isnan(first["meas"][:, 0])
    axz.scatter(first["meas"][m, 0], first["meas"][m, 1], s=6, c="0.7", label="camera meas.")
    axz.plot(first["p_true"][:, 0], first["p_true"][:, 1], "k-", lw=1.5, label="true cube")
    kg = int(T_GRASP / CTRL_DT)
    for key, L in logs.items():
        c, lab = COLORS[key], metrics[key]["label"]
        ax3.plot(*L["x_hat"].T, color=c, lw=1.2, label=f"KF est. – {lab}")
        ax3.plot(*L["tcp"].T, color=c, lw=1.0, ls="--", label=f"TCP – {lab}")
        axz.plot(L["x_hat"][:, 0], L["x_hat"][:, 1], color=c, lw=1.2)
        axz.plot(L["tcp"][:, 0], L["tcp"][:, 1], color=c, ls="--", lw=1.2)
        axz.plot(*L["tcp"][kg, :2], "o", color=c, ms=8, mec="k")
        err = np.linalg.norm(L["x_hat"] - L["p_true"], axis=1)
        axe.plot(t, 1e3 * err, color=c, lw=1.2, label=lab)
        if key != "exact":
            upd = np.arange(1, len(L["gain_err"]) + 1) * CAM_EVERY * CTRL_DT
            axg.semilogy(upd, L["gain_err"], color=c, lw=1, label=lab)
    pg = first["p_true"][kg]
    axz.add_patch(plt.Circle(pg[:2], GRASP_TOL, fill=False, ls=":", color="k"))
    axz.plot(*pg[:2], "kx", ms=10, label="true cube @ grasp")
    axz.set_xlim(pg[0] - 0.06, pg[0] + 0.06)
    axz.set_ylim(pg[1] - 0.14, pg[1] + 0.05)
    axz.set_aspect("equal")
    axz.set_title("Top view near intercept (dashed = TCP, solid = KF; circle = 1.5 cm tolerance)")
    axz.set_xlabel("x [m]"); axz.set_ylabel("y [m]"); axz.legend(fontsize=8, loc="lower right")
    ax3.set_title("3D trajectories (seed 0)")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.view_init(28, -140); ax3.legend(fontsize=7, loc="upper left")
    for a in (axe,):
        a.axvline(T_TRACK, color="0.6", ls=":"); a.axvline(T_GRASP, color="k", ls=":")
        a.set_xlim(0, T_GRASP + 0.1)
    axe.set_ylim(0, 40)
    axe.set_title("KF position error |p̂ − p|  (dotted: approach start, grasp)")
    axe.set_xlabel("t [s]"); axe.set_ylabel("mm"); axe.legend(fontsize=8)
    axg.set_title("Kalman-gain relative error vs exact solve, per update (exact = reference)")
    axg.set_ylim(1e-5, 3)
    axg.set_xlabel("t [s]"); axg.set_ylabel("‖K − K_exact‖ / ‖K_exact‖"); axg.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mc(rows, Ns, irs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["exact"] + [f"thermo_{N}" for N in Ns] + [ir_key(*c) for c in irs]
    labels = ["exact"] + [f"N={N}" for N in Ns] + \
             [f"IR{st}×{N}" + ("w" if w else "") for N, st, w in irs]
    by = {k: [r for r in rows if r["method"] == k] for k in keys}
    col = ["#3388ff"] + list(plt.cm.autumn(np.linspace(0.85, 0.05, len(Ns)))) + \
          list(plt.cm.cool(np.linspace(0.2, 0.9, len(irs))))
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5))
    g = [np.mean([r["gain_err"] for r in by[f"thermo_{N}"]]) for N in Ns]
    ax[0, 0].loglog(Ns, g, "o-", color="#ff8800", label="thermox plain (mean over trials)")
    ax[0, 0].loglog(Ns, g[0] * np.sqrt(Ns[0] / np.array(Ns)), "k:", label="∝ 1/√N")
    for (N, st, w), c in zip(irs, col[1 + len(Ns):]):
        e = np.mean([r["gain_err"] for r in by[ir_key(N, st, w)]])
        ax[0, 0].loglog(st * N, e, "s" if w else "D", color=c, ms=8, mec="k",
                        label=f"IR {st}×{N}" + (" warm-start" if w else " cold"))
    ax[0, 0].set_xlabel("total device samples per gain = steps × N  (≈ hardware integration time)")
    ax[0, 0].set_ylabel("Kalman-gain relative error"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("Solver accuracy vs sample budget (exact solve = reference)")
    for a, met, ttl, ylab in [(ax[0, 1], "pos_rmse", "KF position RMSE (0.5 s → grasp)", "mm"),
                              (ax[1, 0], "grasp_err", "Grasp error at intercept", "mm")]:
        data = [1e3 * np.array([r[met] for r in by[k]]) for k in keys]
        bp = a.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], col):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        a.set_title(ttl); a.set_ylabel(ylab)
        a.tick_params(axis="x", labelsize=8, labelrotation=30)
    ax[1, 0].axhline(1e3 * GRASP_TOL, color="k", ls="--", lw=1)
    ax[1, 0].text(0.6, 1e3 * GRASP_TOL + 0.4, "tolerance 15 mm", fontsize=8)
    succ = [100 * np.mean([r["success"] for r in by[k]]) for k in keys]
    lo, hi = ax[1, 0].get_ylim()
    ax[1, 0].set_ylim(lo, hi + 4)
    for i, s in enumerate(succ):
        ax[1, 0].text(i + 1, hi + 3.6, f"{s:.0f}%", ha="center",
                      va="top", fontsize=9, fontweight="bold")
    ax[1, 0].set_title("Grasp error at intercept (bold: pick success rate)")
    ms = [np.median([r["solve_ms"] for r in by[k]]) for k in keys]
    ax[1, 1].bar(labels, ms, color=col)
    ax[1, 1].tick_params(axis="x", labelsize=8, labelrotation=30)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylabel("ms per KF gain solve (median)")
    ax[1, 1].set_title("Wall-clock on CPU – thermox is a software EMULATION;\n"
                       "not indicative of thermodynamic hardware speed", fontsize=10)
    fig.suptitle(f"Monte Carlo: {len(by['exact'])} seeds per method, identical cube motion "
                 f"and camera noise across methods", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=25, help="Monte Carlo seeds per method")
    ap.add_argument("--out", default="results")
    ap.add_argument("--Ns", default="10,30,100,300,1000,10000")
    ap.add_argument("--ir", default="3x10,1x10w,3x10w,3x33w",
                    help="iterative-refinement configs: STEPSxN, suffix w = warm start; '' = none")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only redraw MC figure from JSON")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Ns = [int(x) for x in a.Ns.split(",")]
    irs = [parse_ir(x) for x in a.ir.split(",") if x]
    if a.replot:
        with open(os.path.join(a.out, "mc_results.json")) as f:
            plot_mc(json.load(f), Ns, irs, os.path.join(a.out, "monte_carlo.png"))
        return
    model = build_model()

    # ---- showcase run (seed 0): exact vs thermox N=10000 vs N=10 vs IR 3x10 warm
    show = [ExactSolver(), ThermoSolver(10000, 0), ThermoSolver(10, 0),
            ThermoRefineSolver(10, 3, True, 0)]
    logs, smet, panels = {}, {}, []
    renderer = None if a.no_video else mujoco.Renderer(model, 330, 400, max_geom=6000)
    for s in show:
        s(np.eye(3), np.ones((3, 6)))     # JIT warm-up
        if hasattr(s, "reset"):
            s.reset()                     # drop warm-start state left by the warm-up
        cb, frames = (None, None) if a.no_video else make_recorder(renderer, COLORS[s.key])
        met, L = run_trial(model, s, seed=0, frame_cb=cb)
        met["label"] = s.label
        logs[s.key], smet[s.key] = L, met
        print("showcase", met, flush=True)
        if frames is not None:
            panels.append(dict(frames=frames, title=s.label, color=COLORS[s.key],
                               grasp_err=met["grasp_err"], every=8))
    plot_showcase(logs, smet, os.path.join(a.out, "trajectories_seed0.png"))
    if panels:
        compose_video(panels, os.path.join(a.out, "pick_place_comparison.gif"))

    if a.mc <= 0:
        return
    # ---- Monte Carlo
    rows = []
    for seed in range(a.mc):
        for s in [ExactSolver()] + [ThermoSolver(N, seed) for N in Ns] + \
                 [ThermoRefineSolver(N, st, w, seed) for N, st, w in irs]:
            met, _ = run_trial(model, s, seed)
            rows.append(met)
        print(f"seed {seed} done", flush=True)
        with open(os.path.join(a.out, "mc_results.json"), "w") as f:
            json.dump(rows, f, indent=1)
    plot_mc(rows, Ns, irs, os.path.join(a.out, "monte_carlo.png"))

    keys = ["exact"] + [f"thermo_{N}" for N in Ns] + [ir_key(*c) for c in irs]
    summary = {}
    for k in keys:
        r = [x for x in rows if x["method"] == k]
        summary[k] = dict(
            success_rate=float(np.mean([x["success"] for x in r])),
            grasp_err_mm_mean=1e3 * float(np.mean([x["grasp_err"] for x in r])),
            pos_rmse_mm=1e3 * float(np.mean([x["pos_rmse"] for x in r])),
            vel_rmse_mm_s=1e3 * float(np.mean([x["vel_rmse"] for x in r])),
            gain_rel_err=float(np.mean([x["gain_err"] for x in r])),
            solve_ms_median=float(np.median([x["solve_ms"] for x in r])))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
