# Development setup

```bash
git clone https://github.com/qml-observer/qml-observer
cd qml-observer
pip install -e ".[dev]"
pre-commit install
```

`[dev]` installs `pytest`, `pytest-cov`, `ruff`, `mypy`, and `pre-commit`.
Add `[pennylane]`/`[qiskit]` (or both) if you're working on adapter code:

```bash
pip install -e ".[dev,pennylane,qiskit]"
```

## Running the test suite

```bash
pytest                          # full suite
pytest tests/unit               # unit tests only
pytest tests/integration         # integration tests only (skipped automatically
                                 # if pennylane/qiskit aren't installed)
pytest --cov=qml_observer        # with coverage
```

## Linting and type-checking

```bash
ruff check .
ruff format --check .
mypy src
```

`pre-commit install` wires both of the above (plus formatting) into a
pre-commit hook so CI failures are caught locally first.

## Running the benchmark suite

```bash
python benchmarks/run_benchmarks.py --seeds 50 --json benchmarks/results/calibration_results.json
```

or open `benchmarks/qml_observer_benchmark.ipynb` for the narrative
version (requires `jupyter`/`ipykernel`, not part of `[dev]` -- install
separately: `pip install jupyter`).

## Repository layout

See `blueprint.md`'s Volume I for the full target repository tree, or
`architecture/overview.md` for a layer-by-layer explanation of
`src/qml_observer/`.
