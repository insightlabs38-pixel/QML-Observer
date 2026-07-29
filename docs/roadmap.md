# Roadmap

See `CHANGELOG.md` for what has already shipped, milestone by milestone.
This page covers what's next, at a high level.

## Near-term (post-0.1)

- Milestone 8: Qiskit adapter, callback integration, examples, tests --
  **already shipped** ahead of the 0.1 release (see `CHANGELOG.md`).
- Milestone 9: gradient SNR, shot-noise-aware `NoiseDetector`, statistical
  confidence intervals -- reducing false positives further, especially for
  finite-shots training. **Complete (Issues #64-#69b)**: gradient SNR/
  measurement-uncertainty primitives, `NoiseDetector`, diagnosis-engine
  noise/plateau priority separation, finite-shots benchmark fixtures,
  gradient-norm confidence intervals surfaced in `BarrenPlateauDetector`
  evidence, and the Issue #69b reconciliation check confirming
  `NoiseDetector` doesn't shift any Milestone 7 number -- see
  `CHANGELOG.md`.
- Milestone 10: alerting (webhooks, Slack-compatible payloads, severity
  levels, deduplication, cooldowns).
- Milestone 11: a live dashboard (loss/gradient charts, diagnosis panel,
  compute-usage panel, run history) -- intentionally *not* built before the
  detection layer is trustworthy (addendum's stated ordering).
- Milestone 12: research-grade diagnostics -- QFIM estimation and
  conditioning, effective rank, parameter-redundancy detection,
  Hessian-vector products, loss-landscape sampling, qubit/depth scaling
  analysis.
- Milestone 13: a recovery engine -- reinitialize parameters, reduce
  circuit depth, switch ansatz/optimizer, adjust learning rate or shot
  budget, natural-gradient methods -- ranked and tested before resuming a
  run automatically. Not implemented until the detection system itself is
  validated (see `docs/research/validation.md`).
- Milestone 14: broader ecosystem support -- PyTorch/JAX hybrid-workflow
  adapters, a generic autograd adapter, experiment-tracker integrations,
  and a documented third-party detector plugin API.
- Milestone 15 (planned, pre-1.0 hardening -- see
  `future_milestones_plan.md`): public API freeze, performance/overhead
  benchmarking, a deliberate thread-safety decision, supply-chain
  hardening, JSONL schema versioning, extended property-based test
  coverage, and a full Definition-of-Done documentation audit. Per that
  plan's own "Gaps & recommendations" section, two items are being pulled
  forward rather than saved for this milestone:
  - **Issue #108 (JSONL schema versioning): done.** Every JSONL record
    (`event`/`diagnosis`/`summary`) now carries a `schema_version` field
    (`reporting/jsonl.py::JSONL_SCHEMA_VERSION`) -- added now, alongside
    Milestone 9's new `GradientSnapshot` CI fields, rather than
    retrofitted onto an already-larger set of historical log shapes later.
  - **Issue #105 (performance/overhead benchmarking): basic version done,
    full soak-test version still pending.** See
    `benchmarks/run_overhead_benchmark.py` for per-step overhead and
    memory-growth numbers on a moderate-length (2000-step) run. Measured
    baseline: ~1,140 steps/sec (~875µs/step) with the full four-detector
    set attached, vs. ~3,870 steps/sec (~257µs/step) with none -- adding
    JSONL logging on top costs a further ~7% (~1,060 steps/sec). No hard
    target is set for v0.1 (same convention as Issue #54); a 100k+-step
    soak test and CI-gated regression tracking remain for the full
    Milestone 15 pass.

## Hardware integration (funding-dependent)

Per addendum §4, real hardware/cloud QPU integration is explicitly out of
scope for the entire `0.x` series. If/when it is funded, it would need:

- Backend credential handling (IBM Quantum, AWS Braket, IonQ, etc.) that
  does not leak into the framework-agnostic core.
- Queue-time telemetry -- how long a job waited vs. how long it ran, since
  "compute saved" should eventually account for queue time too.
- Cost-per-shot estimation per provider, since "estimated compute saved"
  today (Issue #51) is a wall-clock-time estimate, not a dollar estimate.
- Backend noise-profile ingestion, so the noise-aware diagnostics in
  Milestone 9 can distinguish hardware noise from shot noise from a genuine
  plateau on real devices, not just simulators.

`QMLMonitor`'s `shots` parameter and `CircuitMetadata` are already
forward-compatible with this -- no breaking change will be required when
hardware support lands, only additive backend-specific modules.
