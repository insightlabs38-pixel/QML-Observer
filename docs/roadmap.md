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
  levels, deduplication, cooldowns). **Complete (Issues #70-#75c)**:
  `WebhookAction` generic webhook delivery, the structured `AlertPayload`
  shape, a Slack-compatible formatter, severity gating reusing
  `DiagnosisResult.severity`/`SEVERITY_RANK`, change-based alert
  deduplication with an optional time-based cooldown, a `redact_evidence`
  payload-redaction option, and a minimal SSRF safeguard on webhook
  target URLs -- see `docs/integrations/webhook.md`.
- Milestone 11: a live dashboard (loss/gradient charts, diagnosis panel,
  compute-usage panel, run history) -- intentionally *not* built before the
  detection layer is trustworthy (addendum's stated ordering). **Complete:**
  Issues #76-#82b (architecture scaffold, loss chart, gradient chart,
  diagnosis panel, compute-usage panel, run history, data export, and the
  refuse-by-default non-loopback hardening) are implemented as the optional
  `qml-observer[dashboard]` extra -- see `docs/architecture/dashboard.md`.
- Milestone 12: research-grade diagnostics -- QFIM estimation and
  conditioning, effective rank, parameter-redundancy detection,
  Hessian-vector products, loss-landscape sampling, qubit/depth scaling
  analysis. **Complete: Issues #83-#89** (`qml_observer.advanced.geometry`:
  `estimate_qfim`, `qfim_condition_number`/`effective_rank`/
  `summarize_conditioning`, `detect_redundant_parameters`,
  `estimate_hessian_vector_product`, and the `sample_loss_landscape_1d`/
  `_2d`/`random_direction`/`landscape_flatness` loss-landscape sampling
  utilities; `qml_observer.advanced.scaling`: `ScalingAnalyzer.
  analyze_qubit_scaling`/`analyze_depth_scaling` and
  `scaling_observation_from_run_summary`) -- observation-only, opt-in
  research utilities, deliberately not wired into `QMLMonitor`'s per-step
  path (plan.md §26). See `docs/research/geometry.md` for the Issue
  #89b Definition-of-Done writeup (math, references, validation,
  benchmark status, known limitations) per function/issue, including the
  benchmark-corpus and real-circuit-validation gaps flagged as future
  work for #83-#89.
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
