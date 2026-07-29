# Detectors

Every detector implements `BaseDetector` (`detectors/base.py`):
`update(event, state)` accumulates rolling evidence, `diagnose()` returns a
`DetectorResult` (`detector_name`, `triggered`, `confidence`, `evidence`,
`recommendations`), and `reset()` clears internal state for a new run.
Detectors never see each other; the diagnosis engine is solely responsible
for combining their outputs.

MVP detectors (Milestone 4): `BarrenPlateauDetector`, `StagnationDetector`,
`ConvergenceDetector` -- see `docs/detectors/*.md` for what each one
actually checks and how to interpret its evidence. `NoiseDetector`
(gradient SNR-aware, shot-budget-aware) shipped in Milestone 9
(Issue #66).

See `docs/development/adding_detectors.md` for how to add a new one.
