# `NoiseDetector` (not yet implemented)

**Planned module:** `qml_observer.detectors.noise`
**Planned issue:** Milestone 9, Issue #66 (`docs/roadmap.md`)

Will report `IssueType.NOISE_DOMINATED` when gradient signal-to-noise is
too low to trust -- distinguishing "this gradient is genuinely near zero"
from "this gradient estimate is too noisy to tell." The blueprint sketches
an MVP shape (`snr_threshold`, `patience`) that later versions extend with
shot-based uncertainty (`statistics/snr.py`'s
`estimate_measurement_uncertainty()`).

This page will be filled in when Milestone 9 ships; it's listed here now
so the documentation tree matches the architecture even for not-yet-built
pieces, per the blueprint's full repository layout.
