# Roadmap

See `CHANGELOG.md` for what has already shipped, milestone by milestone.
This page covers what's next, at a high level.

## Near-term (post-0.1)

- Milestone 8: Qiskit adapter, callback integration, examples, tests --
  **already shipped** ahead of the 0.1 release (see `CHANGELOG.md`).
- Milestone 9: gradient SNR, shot-noise-aware `NoiseDetector`, statistical
  confidence intervals -- reducing false positives further, especially for
  finite-shots training.
- Milestone 10: alerting (webhooks, Slack-compatible payloads, severity
  levels, deduplication, cooldowns).

## Mid-term

- Milestone 11: a live dashboard (loss/gradient charts, diagnosis panel,
  compute-usage panel, run history) -- intentionally *not* built before the
  detection layer is trustworthy (addendum's stated ordering).
- Milestone 12: research-grade diagnostics -- QFIM estimation and
  conditioning, effective rank, parameter-redundancy detection,
  Hessian-vector products, loss-landscape sampling, qubit/depth scaling
  analysis.

## Long-term

- Milestone 13: a recovery engine -- reinitialize parameters, reduce
  circuit depth, switch ansatz/optimizer, adjust learning rate or shot
  budget, natural-gradient methods -- ranked and tested before resuming a
  run automatically. Not implemented until the detection system itself is
  validated (see `docs/research/validation.md`).
- Milestone 14: broader ecosystem support -- PyTorch/JAX hybrid-workflow
  adapters, a generic autograd adapter, experiment-tracker integrations,
  and a documented third-party detector plugin API.

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
