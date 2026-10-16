# Reusable Launch Vehicle Landing Simulator

A modular 6-DOF simulator for reusable-launch-vehicle descent and landing. A native C++ plant integrates the vehicle dynamics, and swappable Python guidance, navigation, and control modules plug in through Hydra configs.

## Highlights
- Native C++ 6-DOF plant (pybind11): fixed-step RK4 at 1 kHz with throttle/TVC, grid fins, RCS, propellant slosh, and Dryden/von Kármán wind.
- Powered-descent guidance: LCvx (lossless-convexification SOCP on Clarabel) and drag-aware SCvx solved natively by a vendored QOCO solver.
- Multirate closed loop (navigation 100 Hz, control 50 Hz, guidance 1 Hz) with a latching phase scheduler from reentry burn to touchdown.
- Swappable vehicles (Falcon 9 single-engine and 9-engine octaweb, SALTO/Themis T3 with tabulated aero), wind and faults (engine-out, TVC bias, fin lock).
- Seeded multiprocessing Monte Carlo with KPI aggregation; every run exports `replay.json` for a three.js 3D replay viewer.

## Stack
C++20 · pybind11 · Python 3.10+ · NumPy · SciPy · Hydra · CVXPY · Clarabel · QOCO · three.js · pytest

## Layout
```text
src/native/   C++ 6-DOF plant and SCvx solver
src/gnc/      Guidance, navigation, control, allocation
src/sim/      Closed loop, KPIs, Monte Carlo, replay
configs/      Hydra config groups
scripts/      Run, Monte Carlo, viewer entry points
viz/          Three.js replay viewer, Frame Studio
tests/        Pytest suite
docs/         Architecture and module guide
```

## Usage
```bash
python -m pip install -e ".[dev]"   # builds the C++ extension (C++20 compiler)
python scripts/run_single.py vehicle_set=falcon9 plant/actuator=ideal gnc/guidance=scvx
python scripts/run_montecarlo.py vehicle_set=falcon9 plant/actuator=ideal run=montecarlo run.n_episodes=32
python scripts/view_replay.py       # 3D replay of the latest run in results/
python -m pytest tests/ -q
```
