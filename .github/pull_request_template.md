## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] Tests added/updated for the change (unit and/or integration).
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `mypy src` passes.
- [ ] `pytest` passes locally (with `[dev,pennylane,qiskit]` installed, if
      the change touches an adapter).
- [ ] Docs updated (`docs/`, `README.md`, or `CHANGELOG.md`) if
      user-visible behavior changed.
- [ ] If this touches detector thresholds or calibration, the benchmark
      suite (`benchmarks/run_benchmarks.py`) was re-run and results in
      `docs/research/validation.md`/`docs/research/benchmarks.md` updated.

## Related issue

<!-- e.g. Closes #NN -->
