# Kalman-filter pick-and-place: exact solve vs. thermodynamic solve (thermox)

A Franka Panda (MuJoCo) picks a cube off a moving conveyor using a Kalman-filter
tracker. The only difference between methods is how the Kalman gain is computed:

    K^T = S^{-1} (H P),   S = H P H^T + R   (3x3 SPD, 6 right-hand sides)

- **Exact**: `numpy.linalg.solve`
- **Thermodynamic**: `thermox.linalg.solve` (time-average of an Ornstein–Uhlenbeck
  process = software emulation of a thermodynamic linear-algebra chip), with
  `num_samples = N` as the "hardware integration time" budget.

## Setup and run

```bash
./setup.sh                                   # deps + Panda model (MuJoCo Menagerie)
export MUJOCO_GL=osmesa                      # headless Linux (or egl); not needed on desktop
python kf_thermox_pick_place.py --mc 30      # showcase video + figures + 30-seed Monte Carlo
python kf_thermox_pick_place.py --mc 0       # showcase (video + trajectory figure) only
python kf_thermox_pick_place.py --replot     # redraw Monte Carlo figure from results/mc_results.json
```

Options: `--out DIR`, `--Ns 10,30,100,300,1000,10000`, `--ir 3x10,1x10w,3x10w,3x33w`
(iterative-refinement configs, `STEPSxN`, `w` = warm start, `--ir ""` = none), `--no-video`.
The model path can be set with `PANDA_DIR` (default `menag/franka_emika_panda`).

## Test design

| Item | Setting |
|---|---|
| Robot | Franka Panda (Menagerie), physics-stepped at 2 ms, joint position servos, damped-least-squares IK at 100 Hz |
| Object | 4 cm cube on a conveyor, 8 cm/s nominal, velocity random walk (~1.4 cm/s drift) |
| Sensor | Simulated camera, σ = 1.5 cm, 33 Hz |
| Filter | Constant-velocity KF (state p, v), predict 100 Hz, Joseph-form covariance update |
| Planner | Watch 1.5 s, then 2 s minimum-jerk intercept re-planned every tick from the KF prediction |
| Grasp | Success if TCP within 15 mm of true cube centre at intercept (suction-style attach), then lift, transport, drop in bin |
| Fairness | Identical cube motion and camera noise per seed for every method (common random numbers) |
| Monte Carlo | 30 seeds × 7 solver settings |

thermox input scaling (as a real analog device would need): `S` is scaled so its
eigenvalues are ~1 (OU relaxation time ~1), and each right-hand-side column is
normalised to unit norm. Without the column normalisation, gain error at N=10000
was ~6x higher.

## Results (30 seeds per method)

| Solver | Gain rel. error | Pos. RMSE (mm) | Vel. RMSE (mm/s) | Grasp error (mm) | Pick success |
|---|---|---|---|---|---|
| Exact (numpy) | 0 (reference) | 7.86 | 16.9 | 9.68 | 90% |
| thermox N=10000 | 1.8% | 7.86 | 16.9 | 9.66 | 90% |
| thermox N=1000 | 5.6% | 7.88 | 17.2 | 9.71 | 87% |
| thermox N=300 | 10.2% | 7.95 | 17.5 | 9.72 | 87% |
| thermox N=100 | 17.4% | 8.19 | 18.9 | 9.93 | 83% |
| thermox N=30 | 31.1% | 8.59 | 21.7 | 9.73 | 87% |
| thermox N=10 | 50.3% | 10.01 | 31.1 | 11.22 | 80% |

Takeaways:

- Gain error scales as 1/√N, as expected for time-averaging an OU process.
- The KF is tolerant of noisy gains: from N≈1000 up it is indistinguishable from the
  exact solve. Velocity estimates degrade earlier than position estimates.
- Pick-success differences are 1–3 trials out of 30, i.e. not statistically
  significant. Grasp error is dominated by ~5 mm servo tracking lag at intercept.
- Wall-clock times (in `summary.json`) are for the CPU emulation and say nothing
  about thermodynamic hardware speed. thermox even calls an exact solve internally
  to set the OU mean; it models the device's noise, not its speed.

## Improving the thermodynamic solve: warm-started iterative refinement

**Where the plain error comes from.** `thermox.linalg.solve` starts the OU process at
`x0 = 0`, has no burn-in, and averages samples one relaxation time apart under fixed
unit diffusion. With `S` scaled to eig ~1 and unit-norm columns, that gives
sampling noise ≈ √(3.25/N) plus a start-up shrinkage bias ≈ −1.6/(N+1). This matches the
measured 1.8 % at N=10000 and 5.6 % at N=1000. The noise is set by the device physics,
so a bigger N is the only lever for a single-shot solve.

**Fix (`ThermoRefineSolver`).** The device solve is wrapped in digital mixed-precision
iterative refinement, as used for low-precision analog in-memory solvers:

    X <- previous gain (warm) or 0 (cold)
    repeat steps times:   X <- X + device_solve(S, B - S X)   # residual rescaled to full range

- Refinement: each step the device solves only for the correction, re-normalised to its
  full range. The error therefore shrinks by the device's relative error ε(N) per step:
  ε^steps instead of 1/√N. Measured: 50 % → 25 % → 12.5 % for N=10.
- Warm start: the Kalman gain changes little between camera frames, so the first residual
  is already small and the device only resolves that change. This works like an
  analog device that keeps its state between solves.
- Device physics, noise level and scaling are unchanged. The digital side only adds a
  residual matvec (O(d²)) per step. Budget = steps × N device samples per gain.

**Results (100 seeds, same CRN seeds as above; `results_ir/`).** Δvel is the paired
difference in velocity RMSE vs exact (± standard error). "≠ exact" counts seeds whose
pick outcome differs from the exact solver's.

| Solver | Samples / gain | Gain rel. error | Pos. RMSE (mm) | Vel. RMSE (mm/s) | Δvel vs exact | Grasp err (mm) | Pick success | ≠ exact |
|---|---|---|---|---|---|---|---|---|
| Exact (numpy) | – | 0 | 8.13 | 17.24 | – | 8.51 | 94 % | – |
| thermox N=10 | 10 | 50.3 % | 10.35 | 32.22 | +14.99 ± 0.65 | 9.82 | 89 % | 7 |
| thermox N=30 | 30 | 31.1 % | 8.97 | 22.43 | +5.19 ± 0.31 | 8.87 | 92 % | 2 |
| thermox N=100 | 100 | 17.5 % | 8.39 | 19.01 | +1.77 ± 0.18 | 8.61 | 91 % | 3 |
| thermox N=1000 | 1000 | 5.6 % | 8.15 | 17.49 | +0.26 ± 0.06 | 8.54 | 92 % | 2 |
| thermox N=10000 | 10000 | 1.8 % | 8.13 | 17.25 | +0.02 ± 0.02 | 8.50 | 94 % | 0 |
| IR 3×10 cold | 30 | 12.5 % | 8.26 | 18.07 | +0.83 ± 0.13 | 8.61 | 94 % | 0 |
| IR 1×10 warm | 10 | 2.5 % | 8.18 | 17.59 | +0.36 ± 0.08 | 8.51 | 94 % | 0 |
| **IR 3×10 warm** | **30** | **0.47 %** | 8.13 | 17.24 | +0.01 ± 0.02 | 8.51 | 94 % | 0 |
| IR 3×33 warm | 99 | 0.10 % | 8.13 | 17.24 | 0.00 ± 0.00 | 8.51 | 94 % | 0 |

- With a budget of 30 samples, warm-started refinement is 3.8× more accurate than plain thermox with 10000
  samples (~330× less device time), and the robot behaves like the exact solver on every seed.
- Warm start alone (1×10, one device call) beats plain N=1000.
- The task ceiling is the exact solver: the remaining 6 % misses also happen with an exact
  gain (servo lag and estimation error, not the solver).

**Stress test: ill-conditioned `S`.** In this KF, `S` is exactly a scaled identity (isotropic
camera noise, decoupled axes), which is the easiest possible case for the device. With
depth-camera noise 3× or 10× larger along an oblique viewing ray, `S` becomes a full
3×3 matrix. Gain relative error, KF-only, 20 seeds:

| cond(S) | N=100 | N=10000 | IR 3×10 cold | IR 3×10 warm | IR 3×33 warm |
|---|---|---|---|---|---|
| ~9 | 38.7 % | 4.2 % | 27 % | 0.88 % | 0.19 % |
| ~94 | 98 % | 23 % | 82 % | 6.3 % | 2.0 % |

Under ill-conditioning ε(N) approaches 1 (slow OU modes barely relax within N), so cold
refinement stalls. Warm start is what keeps the method working.

**Caveats.**
- The gains are in device time. CPU wall-clock for the emulation says nothing about hardware.
- Each refinement step is a digital↔analog round trip (load RHS, integrate, read out).
  For d=3 the digital residual costs about as much as solving outright; the trade pays
  off for large systems (O(d²) residual vs O(d³) solve).
- Refinement needs ε(N) < 1 per step. For ill-conditioned systems, preconditioning is the
  next lever.
- thermox models no DAC/ADC quantisation or device mismatch. Rescaling residuals to full
  range assumes these stay small relative to ε.

## Files

- `kf_thermox_pick_place.py` – full experiment (scene, IK, KF, solvers, planner, rendering, plots)
- `results_ir/` – same outputs with the iterative-refinement solvers, 100 seeds; video adds an
  "IR 3×10 warm" panel
- `BENCHMARK.md`, `thermo_benchmark.py`, `thermo_benchmark_report.py`, `results_bench/` – IR vs
  plain thermox vs digital solvers on 17 problem sets (KF, IK, MPC, heat equation, random SPD),
  hardware imperfections and a wall-clock model. The benchmark scales the device more
  strictly (top eigenvalue = 1, per-DAC input range), so its numbers differ from the
  stress-test table above.
- `results/pick_place_comparison.gif|.mp4` – side-by-side sim: exact | thermox N=10000 | thermox N=10
- `results/trajectories_seed0.png` – 3D/top-view trajectories, error vs time, gain error per update
- `results/monte_carlo.png` – accuracy vs N, RMSE, grasp error + success, timing
- `results/mc_results.json` – per-trial metrics; `results/summary.json` – aggregates

## Limitations

- Grasp is modelled as attach-within-tolerance, not contact-rich finger physics.
- The cube is kinematic (mocap body); belt, bin and cube have no collisions with the arm.
- Single small linear system (3x3); larger systems (e.g. full-state SLAM or MPC) are
  where thermodynamic hardware is argued to pay off.
