"""Tables, wall-clock model and figures for thermo_benchmark.py results."""
import os
import glob
import json

import numpy as np

TARGET = 1e-4                       # accuracy used for the headline comparison
REACHED = 0.95                      # a config counts if >= 95% of solves hit the target

ROBOT = ["kf_iso", "kf_depth3", "kf_depth10", "ik", "mpc", "mpc_hard"]
CLASSIC = ["heat8", "heat16"]
RANDOM = [f"rand_d{d}_k{k}" for d in (16, 64, 256) for k in (10, 100, 1000)]
LABELS = {
    "kf_iso": "KF gain, isotropic camera", "kf_depth3": "KF gain, depth noise 3×",
    "kf_depth10": "KF gain, depth noise 10×", "ik": "Arm IK (DLS)",
    "mpc": "Torque MPC", "mpc_hard": "Torque MPC, low torque weight",
    "heat8": "Heat eq. 8×8 grid", "heat16": "Heat eq. 16×16 grid",
}
for _n in RANDOM:
    _d, _k = _n[6:].split("_k")
    LABELS[_n] = f"Random SPD d={_d}, κ={_k}"

# Hardware model (assumptions and sources in BENCHMARK.md). Times in seconds.
def cpu_flops(d):
    """Effective float64 rate of one x86 AVX2 core on size-d dense kernels: ~1 GFLOP/s
    at d<=6, ~10-20 at d=100-256 (measured small-Cholesky rates, arXiv:1704.02457)."""
    return 30e9 * d / (d + 100)


def matvec_flops(d):
    """Matrix-vector products are memory-bound once A (8 d^2 bytes) leaves the ~8 MB
    cache: 2 flops per 8 bytes at ~20 GB/s."""
    return cpu_flops(d) if 8 * d * d < 8e6 else min(cpu_flops(d), 20e9 / 4)


CPU_T_CALL = 0.2e-6
HW = {   # tau: fastest relaxation time; t_round: DAC load + ADC read + link, per round
    "proto": dict(label="thermo IR, prototype-class device", tau=0.5e-6, t_round=50e-6,
                  t_prog=lambda d: d * d * 1e-6),
    "chip": dict(label="thermo IR, optimistic integrated chip", tau=10e-9, t_round=200e-9,
                 # row-parallel DACs, but d^2 16-bit entries still cross a 50 Gb/s link
                 t_prog=lambda d: max(d * 10e-9, 2 * d * d / 6.25e9)),
    "limit": dict(label="thermo IR, physical-limit device (1 ns, free I/O)", tau=1e-9,
                  t_round=0.0, t_prog=lambda d: 0.0),
}
POWER_ITERS = 5                     # warm-started power iterations to estimate eig_max

# chart ink + fixed method colours (reference palette, light mode, slot order)
INK, INK2, MUTED, GRID, AXIS, SURF = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                      "#c3c2b7", "#fcfcfb")
COL = {"cg": "#2a78d6", "plain": "#eb6834", "ir_cold": "#1baf7a", "ir_warm": "#eda100"}


def load(out):
    res = {}
    for f in glob.glob(os.path.join(out, "*.json")):
        name = os.path.basename(f)[:-5]
        if not name.startswith("nonideal_") and name != "report":
            with open(f) as fh:
                res[name] = json.load(fh)
    return res


def _tkey(t):
    return f"{t:g}"


def ir_configs(r, tgt, warm):
    """All IR configs that reliably reach tgt: list of dicts with N, prec, rounds."""
    w = "warm" if warm else "cold"
    out = []
    for k, v in r["ir"].items():
        t, ww, N, prec = k.split("|")
        if t == _tkey(tgt) and ww == w and v["reached"] >= REACHED and v["failed"] == 0:
            out.append(dict(N=int(N), prec=prec, rounds=v["rounds"], warm=warm,
                            samples=v["rounds"] * int(N)))
    return out


def best_ir(r, tgt, warm):
    c = ir_configs(r, tgt, warm)
    return min(c, key=lambda x: x["samples"]) if c else None


def best_cg(r, tgt):
    best = None
    for k, v in r["cg"].items():
        t, w, kind = k.split("|")
        if t == _tkey(tgt) and v["reached"] >= REACHED and (best is None or
                                                           v["iters"] < best["iters"]):
            best = dict(iters=v["iters"], warm=w == "warm", kind=kind)
    return best


def plain_needed(r, tgt):
    """Samples plain thermox needs for mean error <= tgt (1/sqrt(N) extrapolation
    from the largest N when not reached). Returns (N, extrapolated)."""
    best = None
    for prec in ("none", "jac"):
        Ns = sorted({int(k.split("|")[0]) for k in r["plain"]})
        errs = [r["plain"][f"{N}|{prec}"]["err"] for N in Ns]
        hit = [N for N, e in zip(Ns, errs) if e <= tgt]
        cand = (hit[0], False) if hit else (Ns[-1] * (errs[-1] / tgt) ** 2, True)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


# ------------------------------------------------------------------ wall clock
def wall_clock(r, tgt):
    """Estimated seconds per solve: Cholesky, best CG, best thermo IR per hardware."""
    f_new = r["n_matrices"] / r["n_systems"]          # share of solves with a new matrix
    cfgs = [c for warm in ([False, True] if r["seq"] else [False])
            for c in ir_configs(r, tgt, warm)]
    cg = best_cg(r, tgt)
    t = model_times(r["d"], r["m"], f_new, cg["iters"], cfgs)
    N_plain, t["plain_extrap"] = plain_needed(r, tgt)
    for hw, v in plain_times(r["d"], r["m"], f_new, N_plain).items():
        t["plain_" + hw] = v
    return t


def plain_times(d, m, f_new, N):
    """Seconds per solve for plain thermox: one chip run of N samples per right-hand side,
    with the same matrix setup and one round trip as IR, and no residual checks."""
    Fmv = matvec_flops(d)
    out = {}
    for hw, h in HW.items():
        setup = f_new * (h["t_prog"](d) + (2 * POWER_ITERS + 3) * d * d / Fmv)
        out[hw] = CPU_T_CALL + setup + m * N * h["tau"] + h["t_round"]
    return out


def model_times(d, m, f_new, cg_iters, ir_cfgs):
    """Seconds per solve for Cholesky, CG and thermo IR (best config) on each device."""
    F, Fmv = cpu_flops(d), matvec_flops(d)
    t = dict(chol=CPU_T_CALL + f_new * d**3 / 3 / F + 2 * d * d * m / Fmv,
             cg=CPU_T_CALL + cg_iters * m * (2 * d * d + 10 * d) / Fmv)
    for hw, h in HW.items():
        best = None
        setup = f_new * (h["t_prog"](d) + (2 * POWER_ITERS + 3) * d * d / Fmv)
        for c in ir_cfgs:
            per_round = m * c["N"] * h["tau"] + h["t_round"] + m * (2 * d * d + 5 * d) / Fmv
            tt = CPU_T_CALL + setup + c["rounds"] * per_round
            if best is None or tt < best[0]:
                best = (tt, c)
        t[hw] = best[0] if best else np.nan
        t[hw + "_cfg"] = best[1] if best else None
    return t


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_cost(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in ROBOT + CLASSIC + RANDOM if n in res]
    y = np.arange(len(names))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.42 * len(names) + 1.8), sharey=True)
    fig.patch.set_facecolor(SURF)
    for i, n in zip(y, names):
        r = res[n]
        Np, ext = plain_needed(r, TARGET)
        a1.plot(Np, i + 0.18, "o", ms=8, mfc="none" if ext else COL["plain"],
                mec=COL["plain"], mew=2)
        for warm, key, dy in ((False, "ir_cold", 0.0), (True, "ir_warm", -0.18)):
            if warm and not r["seq"]:
                continue
            b = best_ir(r, TARGET, warm)
            if b:
                a1.plot(b["samples"], i + dy, "o", ms=8, color=COL[key], mec=SURF, mew=1.5)
            else:
                a1.plot(3e12, i + dy, "x", ms=8, color=COL[key], mew=2)
        cg = best_cg(r, TARGET)
        ir = [b for b in (best_ir(r, TARGET, False),
                          best_ir(r, TARGET, True) if r["seq"] else None) if b]
        a2.plot(cg["iters"], i + 0.1, "o", ms=8, color=COL["cg"], mec=SURF, mew=1.5)
        if ir:
            bb = min(ir, key=lambda x: x["rounds"])
            a2.plot(bb["rounds"], i - 0.1, "o", ms=8,
                    color=COL["ir_warm" if bb["warm"] else "ir_cold"], mec=SURF, mew=1.5)
        else:
            a2.plot(60, i - 0.1, "x", ms=8, color=COL["ir_cold"], mew=2)
    for a in (a1, a2):
        _style(a)
        a.set_xscale("log")
    a1.set_yticks(y)
    a1.set_yticklabels([LABELS[n] + ("" if n in RANDOM else
                                     f"  (d={res[n]['d']}, κ≈{res[n]['cond_median']:.2g})")
                        for n in names], color=INK, fontsize=8.5)
    a1.set_xlim(3, 1e13)
    a1.set_xlabel("device time per solve [samples = fastest device time constants]",
                  color=INK2)
    a1.set_title(f"Thermodynamic device time to reach {TARGET:g} relative error",
                 color=INK, loc="left", fontsize=11)
    a2.set_xlim(0.7, 100)
    a2.set_xlabel("O(d²) matrix-vector products per solve on the digital side", color=INK2)
    a2.set_title("Digital work to reach the same accuracy", color=INK, loc="left",
                 fontsize=11)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], ls="", marker="o", ms=8, color=COL["plain"], label="plain thermox"),
          Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=COL["plain"], mew=2,
                 label="plain thermox (1/√N extrapolated)"),
          Line2D([], [], ls="", marker="o", ms=8, color=COL["ir_cold"], label="IR, cold start"),
          Line2D([], [], ls="", marker="o", ms=8, color=COL["ir_warm"], label="IR, warm start"),
          Line2D([], [], ls="", marker="x", ms=8, mew=2, color=MUTED,
                 label="never reached (stalls)")]
    a1.legend(handles=h1, fontsize=8, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.45, -0.07), labelcolor=INK2)
    h2 = [Line2D([], [], ls="", marker="o", ms=8, color=COL["cg"],
                 label="digital CG / Jacobi-PCG (iterations)"),
          Line2D([], [], ls="", marker="o", ms=8, color=COL["ir_warm"],
                 label="thermo IR (rounds; colour = cold/warm)")]
    a2.legend(handles=h2, fontsize=8, frameon=False, loc="upper center", ncol=1,
              bbox_to_anchor=(0.5, -0.07), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def fig_scaling(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [10, 100, 1000]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURF)
    seq = ["#2a78d6", "#eb6834", "#1baf7a"]          # d = 16, 64, 256 (fixed order)
    for d, c in zip((16, 64, 256), seq):
        s, it = [], []
        for k in ks:
            r = res.get(f"rand_d{d}_k{k}")
            b = best_ir(r, TARGET, False) if r else None
            s.append(b["samples"] if b else np.nan)
            it.append(best_cg(r, TARGET)["iters"] if r else np.nan)
        a1.plot(ks, s, "-o", color=c, lw=2, ms=7, mec=SURF, label=f"d = {d}")
        a2.plot(ks, it, "-o", color=c, lw=2, ms=7, mec=SURF, label=f"d = {d}")
    for a in (a1, a2):
        _style(a)
        a.grid(True, axis="y", color=GRID, lw=0.6)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("condition number κ", color=INK2)
        a.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    a1.set_ylabel("device samples per solve", color=INK2)
    a1.set_title("Thermo IR (cold): device time grows with κ, not d", color=INK, loc="left",
                 fontsize=10.5)
    a2.set_ylabel("CG iterations per solve", color=INK2)
    a2.set_title("Digital CG: iterations grow with √κ and are capped near d", color=INK,
                 loc="left", fontsize=10.5)
    fig.suptitle(f"Random dense SPD systems, target {TARGET:g}", color=INK2, fontsize=9,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def fig_nonideal(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(out, "nonideal_*.json")))
    if not files:
        return
    order = ["kf_iso", "ik", "heat8", "rand_d64_k10", "mpc", "rand_d64_k1000"]
    runs = sorted((json.load(open(f)) for f in files),
                  key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.patch.set_facecolor(SURF)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    sweeps = (("mismatch", "coupling mismatch (relative std)"),
              ("bits", "converter + coupling resolution [bits]"))
    for row, (kind, xlab) in enumerate(sweeps):
        a_err, a_rnd = axes[row]
        for run, c in zip(runs, cols):
            ks = [k for k in run["sweeps"] if k.startswith(kind)]
            xs = [float(k.split("=")[1]) for k in ks]
            xs = [x if x > 0 else 0.003 for x in xs] if kind == "mismatch" else xs
            rows = [run["sweeps"][k] for k in ks]
            ok = [r["unstable"] == 0 for r in rows]
            a_err.plot(xs, [r["plain_err"] if o else np.nan for r, o in zip(rows, ok)],
                       "-o", color=c, lw=2, ms=6, mec=SURF,
                       label=f"{LABELS[run['name']]}  (κ≈{run['cond_median']:.2g}, "
                             f"IR N={run['N_ir']})")
            a_rnd.plot(xs, [r["ir_rounds"] if o else np.nan for r, o in zip(rows, ok)],
                       "-o", color=c, lw=2, ms=6, mec=SURF)
            for a in (a_err, a_rnd):
                bad = [x for x, o in zip(xs, ok) if not o]
                if bad:
                    a.plot(bad, [a is a_err and 3.0 or 45] * len(bad), "x", color=c, ms=9,
                           mew=2.5)
        for a in (a_err, a_rnd):
            _style(a)
            a.grid(True, axis="y", color=GRID, lw=0.6)
            a.set_xlabel(xlab, color=INK2)
            if kind == "mismatch":
                a.set_xscale("log")
                a.set_xticks([0.003, 0.01, 0.03, 0.1])
                a.set_xticklabels(["0", "1%", "3%", "10%"])
        a_err.set_yscale("log")
        a_err.set_ylim(0.02, 5)
        a_err.set_ylabel("plain thermox (N=1000) median rel. error", color=INK2)
        a_rnd.set_ylim(0, 50)
        a_rnd.set_ylabel(f"IR rounds to reach {runs[0]['target']:g}", color=INK2)
    axes[0, 0].set_title("Plain thermox: imperfections set an error floor", color=INK,
                         loc="left", fontsize=10.5)
    axes[0, 1].set_title("IR: same imperfections cost only a few extra rounds", color=INK,
                         loc="left", fontsize=10.5)
    from matplotlib.lines import Line2D
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], ls="", marker="x", ms=9, mew=2.5, color=MUTED))
    labels.append("programmed matrix not positive definite: the physical process diverges")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def fig_wallclock(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names = [n for n in ROBOT + CLASSIC + RANDOM if n in res]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(names) + 1.8))
    fig.patch.set_facecolor(SURF)
    marks = [("chol", INK2, "s", "digital Cholesky (exact)"),
             ("cg", COL["cg"], "o", "digital CG / PCG"),
             ("proto", COL["ir_cold"], "D", HW["proto"]["label"]),
             ("chip", COL["ir_warm"], "D", HW["chip"]["label"]),
             ("limit", "#e87ba4", "D", HW["limit"]["label"])]
    for i, n in zip(y, names):
        t = wall_clock(res[n], TARGET)
        for j, (k, c, mk, _) in enumerate(marks):
            v = t[k]
            if np.isfinite(v):
                ax.plot(v * 1e6, i + 0.28 - 0.14 * j, mk, ms=6.5, color=c, mec=SURF)
            else:
                ax.plot(2e6, i + 0.28 - 0.14 * j, "x", ms=6.5, color=c, mew=2)
    _style(ax)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[n] for n in names], color=INK, fontsize=8.5)
    ax.set_xlabel(f"estimated time per solve [µs] to reach {TARGET:g}", color=INK2)
    ax.set_title("Wall-clock model: digital CPU vs thermodynamic device + digital refinement",
                 color=INK, loc="left", fontsize=11)
    ax.legend(handles=[Line2D([], [], ls="", marker=mk, ms=7, color=c, label=lab)
                       for _, c, mk, lab in marks] +
              [Line2D([], [], ls="", marker="x", ms=7, mew=2, color=MUTED,
                      label="thermo never reaches target")],
              fontsize=8, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.4, -0.06), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def fig_size_scaling(res, path, ds=(16, 64, 256, 1024, 4096, 16384)):
    """Time per one-off dense solve vs d. Measured IR rounds / CG iterations at d<=256
    (random SPD); beyond 256 they are held at their d=256 values (extrapolation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    kinds = [("chol", INK2, "digital Cholesky"), ("cg", COL["cg"], "digital CG / PCG"),
             ("proto", COL["ir_cold"], HW["proto"]["label"]),
             ("chip", COL["ir_warm"], HW["chip"]["label"]),
             ("limit", "#e87ba4", HW["limit"]["label"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.patch.set_facecolor(SURF)
    for a, kappa in zip(axes, (10, 100, 1000)):
        series = {k: [] for k, _, _ in kinds}
        for d in ds:
            r = res.get(f"rand_d{min(d, 256)}_k{kappa}")
            cg = best_cg(r, TARGET)["iters"]
            t = model_times(d, 1, 1.0, cg, ir_configs(r, TARGET, False))
            for k in series:
                series[k].append(t[k])
        for k, c, _ in kinds:
            v = np.array(series[k]) * 1e6
            meas = np.array(ds) <= 256
            a.plot(np.array(ds)[meas], v[meas], "-o", color=c, lw=2, ms=6, mec=SURF)
            a.plot(np.array(ds)[~meas | (np.array(ds) == 256)],
                   v[~meas | (np.array(ds) == 256)], "--", color=c, lw=1.5)
        _style(a)
        a.grid(True, axis="y", color=GRID, lw=0.6)
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xticks(ds)
        a.set_xticklabels([str(d) for d in ds])
        a.axvspan(256, ds[-1] * 1.2, color=GRID, alpha=0.35, lw=0)
        a.set_xlim(ds[0] / 1.2, ds[-1] * 1.2)
        a.set_xlabel("matrix size d", color=INK2)
        a.set_title(f"κ = {kappa}", color=INK, loc="left", fontsize=10.5)
    axes[0].set_ylabel(f"time per one-off solve to {TARGET:g} [µs]", color=INK2)
    axes[0].text(300, axes[0].get_ylim()[1] / 3, "extrapolated →", color=INK2, fontsize=8)
    axes[0].legend(handles=[Line2D([], [], color=c, lw=2, label=lab) for _, c, lab in kinds],
                   frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Where would a thermodynamic device win? Dense SPD solve, new matrix every "
                 "solve (solid = measured rounds/iterations, dashed = held at d=256 values)",
                 color=INK2, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ tables
def table(res):
    rows = []
    for n in ROBOT + CLASSIC + RANDOM:
        if n not in res:
            continue
        r = res[n]
        Np, ext = plain_needed(r, TARGET)
        bc, bw = best_ir(r, TARGET, False), best_ir(r, TARGET, True) if r["seq"] else None
        cg = best_cg(r, TARGET)
        t = wall_clock(r, TARGET)
        rows.append(dict(
            name=n, label=LABELS[n], d=r["d"], m=r["m"], cond=r["cond_median"], seq=r["seq"],
            plain_N=Np, plain_extrap=ext,
            ir_cold=bc, ir_warm=bw, cg=cg,
            plain_err_1e4=min(r["plain"][f"10000|{p}"]["err"] for p in ("none", "jac")),
            **{f"t_{k}": t.get(k) for k in ("chol", "cg", *HW)}))
    return rows


def _fmt_ir(b, applicable=True):
    if not applicable:
        return "–"
    if not b:
        return "stalls"
    return f"{b['samples']:.0f} ({b['rounds']:.1f}×N={b['N']})"


def markdown(rows):
    lines = ["| Problem | d | κ | Plain thermox | IR cold | IR warm | Digital CG iters |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = f"{r['plain_N']:.2g}" + (" (extrap.)" if r["plain_extrap"] else "")
        lines.append(f"| {r['label']} | {r['d']} | {r['cond']:.3g} | {p} | "
                     f"{_fmt_ir(r['ir_cold'])} | {_fmt_ir(r['ir_warm'], r['seq'])} | "
                     f"{r['cg']['iters']:.1f} ({'warm ' if r['cg']['warm'] else ''}"
                     f"{r['cg']['kind']}) |")
    return "\n".join(lines)


def markdown_time(rows):
    lines = ["| Problem | Cholesky | CG | " + " | ".join(h["label"] for h in HW.values()) +
             " |", "|---|---|---|" + "---|" * len(HW)]
    f = (lambda v: "never" if v is None or not np.isfinite(v) else
         (f"{v * 1e6:.3g} µs" if v < 1e-3 else f"{v * 1e3:.3g} ms"))
    for r in rows:
        lines.append(f"| {r['label']} | {f(r['t_chol'])} | {f(r['t_cg'])} | " +
                     " | ".join(f(r[f't_{k}']) for k in HW) + " |")
    return "\n".join(lines)


def report(out):
    res = load(out)
    rows = table(res)
    print(markdown(rows))
    fig_cost(res, os.path.join(out, "cost_to_accuracy.png"))
    fig_scaling(res, os.path.join(out, "scaling_random_spd.png"))
    fig_nonideal(out, os.path.join(out, "nonidealities.png"))
    print()
    print(markdown_time(rows))
    fig_wallclock(res, os.path.join(out, "wall_clock.png"))
    fig_size_scaling(res, os.path.join(out, "size_scaling.png"))
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
