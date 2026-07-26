# Methodology

QML Observer's diagnosis engine makes a falsifiable claim every time it
reports `POSSIBLE_BARREN_PLATEAU`: that the observed gradient/loss signals
are consistent with a plateau-like failure mode. The project's third
architectural rule (blueprint) is to make that claim testable, not just
assumed: **explicitly benchmark false positives and false negatives rather
than assuming every near-zero gradient is a barren plateau.**

## Fixtures

All calibration and benchmarking here uses the seeded synthetic-run
generators in `tests/fixtures/synthetic_runs.py` (Milestone 4, Issue #32),
not real quantum circuits, for two reasons:

1. **Determinism.** A seeded fixture produces the exact same
   loss/gradient sequence every run, so a threshold change's effect is
   isolated from simulator/shot-noise variance across runs.
2. **Coverage.** Five scenarios are cheap to generate at scale (hundreds
   of seeds) in a way that running hundreds of real variational circuits
   to convergence is not: `healthy_learning`, `convergence`,
   `artificial_plateau`, `noise_dominated`, `stagnant_optimizer`.

The `depth_scaling` case from plan.md §15 (showing detector suspicion grow
with qubit/depth scaling) is out of scope until `ScalingAnalyzer`
(Milestone 12) exists to generate genuinely scaling circuit families --
see `docs/roadmap.md`.

The real-circuit "critical MVP demo" (an engineered RZ-only/RY-CNOT-RY
plateau ansatz vs. a healthy `efficient_su2` ansatz) lives separately in
`examples/{pennylane,qiskit}/barren_plateau_demo.py` and
`benchmarks/qml_observer_benchmark.ipynb` -- it demonstrates the
end-to-end behavior on live PennyLane/Qiskit circuits, complementing (not
replacing) the synthetic-fixture calibration below.

## Metrics

- **False-positive rate**: across N seeded runs of a fixture that should
  *never* be flagged as a plateau (`healthy_learning`, `convergence`,
  `noise_dominated`), the fraction where `POSSIBLE_BARREN_PLATEAU` was
  ever reported. Target: **< 5%** (addendum §3).
- **Detection rate / latency**: across N seeded runs of
  `artificial_plateau` (which *should* be flagged), the fraction detected
  at all, and the steps-to-first-flag distribution (median, p95). No hard
  target for `0.1` -- baseline first, per addendum §3.

## Process

1. Run `benchmarks/run_benchmarks.py --seeds 50` (or
   `run_calibration_sweep()` for a specific parameter) to get both metrics
   for a given detector configuration.
2. Choose the default that minimizes false positives on
   healthy/convergence/noise fixtures while maximizing true-positive
   detection speed on the artificial-plateau fixture.
3. Document the exact procedure and resulting numbers in
   `research/validation.md` so the choice is reproducible and falsifiable.
4. Any future change to a shipped default threshold is a versioned event:
   a `CHANGELOG.md` entry plus a note in `research/benchmarks.md`
   explaining what changed and why (addendum §3).

This intentionally does **not** use machine learning or automated
hyperparameter search for the MVP -- thresholds are chosen by inspecting
the sweep results directly, keeping the calibration process itself
explainable.
