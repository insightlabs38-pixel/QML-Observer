# `NoiseDetector`

**Module:** `qml_observer.detectors.noise`
**Reports:** `IssueType.NOISE_DOMINATED` (via the diagnosis engine)

Milestone 9, Issue #66. Distinguishes a gradient estimate that is *too
noisy/under-sampled to trust yet* from one that is genuinely small
(`BarrenPlateauDetector`'s concern) -- see that page's "What this is not"
section for the flip side of this same distinction.

## Algorithm

Per step with shot-count information available:

```
uncertainty = estimate_measurement_uncertainty(gradient.variance, shots)
snr         = estimate_gradient_snr(gradient.mean_abs, uncertainty)

IF snr <= snr_threshold
AND condition persists for `patience` consecutive shot-bearing steps
THEN
    increase noise-dominated confidence
```

`gradient.mean_abs` (a *per-parameter* magnitude) is compared against a
per-parameter shot-noise floor derived from `gradient.variance` and the
reported `shots` -- **not** the aggregate `gradient.norm_l2`, which scales
with `sqrt(n_parameters)` and would make the ratio insensitive to shot
count for any circuit with more than a handful of parameters. See the
`detectors/noise.py` module docstring for the full derivation.

**Steps without shot-count information (`shots is None` or `shots <= 0`,
e.g. analytic/adjoint execution) are skipped entirely** -- they neither
extend nor reset the detector's persistence streak. This means
`NoiseDetector` never fires at all on a purely analytic run, which is
correct: "shot noise" has no meaning there, and a collapsed analytic
gradient is unambiguously `BarrenPlateauDetector`'s finding.

## Why this doesn't get conflated with a genuine plateau

A truly collapsed gradient (a real barren plateau) has both a small mean
magnitude *and* a small variance, so the shot-noise floor shrinks right
along with the signal and the ratio stays informative regardless of shot
count -- `NoiseDetector` correctly stays quiet in that case. It only
trips when the magnitude is small *relative to how few shots estimated
it*, which is the actual ambiguity it exists to flag.

The diagnosis engine additionally gives `NOISE_DOMINATED` priority over
`POSSIBLE_BARREN_PLATEAU` (though not over `CONVERGED`) whenever both
trigger for the same step (Issue #67, `diagnosis/scoring.py`), as a
second line of defense on top of the detector-level separation above.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `snr_threshold` | `1.0` | SNR at or below which a step's gradient estimate is considered statistically unreliable. Placeholder per addendum §3 -- see `research/validation.md`'s Milestone 9 section for the shot-budget sweep this was checked against. |
| `patience` | `50` | Consecutive shot-bearing steps the low-SNR condition must hold before triggering. |

## Evidence fields

```
Latest gradient SNR: 0.850 (threshold 1.000).
Latest shot-noise uncertainty estimate: 1.2e-02.
Low-SNR condition has persisted for 62 consecutive shot-bearing step(s) (patience 50).
Observed over 140 step(s) with shot-count information.
```

## What this is *not*

This detector never claims the gradient is actually large -- only that
the current shot budget cannot rule out shot noise as the explanation for
what's observed. Its recommendation is to increase the shot budget or
switch to analytic/adjoint differentiation, then re-check
`BarrenPlateauDetector`'s own reading once a more trustworthy gradient
estimate is available.
