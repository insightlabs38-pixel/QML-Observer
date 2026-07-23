"""Rolling-window statistics primitive.

Milestone 3, Volume IV (`statistics/rolling.py`), Issue #23.

`RollingWindow` is a generic, metric-agnostic bounded history used by
detectors (Milestone 4) to track any scalar time series -- gradient
norms, loss values, update magnitudes, etc. -- and query its mean,
variance, and linear trend without re-deriving persistence bookkeeping
in every detector.

This is *not* `core.state.RunState` (Milestone 2, Issue #12), which
holds full `StepObservation`s for the monitor itself; `RollingWindow`
holds bare scalars for a single tracked metric and lives one layer
below, inside the statistics engine.

Incremental design (per blueprint: "use incremental calculations where
practical"): running sums are maintained across `append()` calls so
`mean()`/`variance()` are O(1) rather than O(window_size) in the common
case. The one deliberate exception is `slope()`, which recomputes a
least-squares fit over the current window on each call -- an exact
incremental sliding-window regression is significantly more complex to
maintain correctly, and `window_size` is bounded (detectors use windows
on the order of 100 steps), so an O(window_size) fit is cheap enough to
not be worth that complexity.

Numerical edge cases (addendum §7): a window with zero or one
observation has an undefined variance/slope -- both return `None`
(explicitly distinguished from `nan`, which is reserved for "there is
enough data, but the data itself contains a non-finite value").
NaN/Inf values are valid entries (a diverging metric is signal, not
an error) and are never rejected by `append()`. Care is taken so that a
NaN/Inf value that *enters and later leaves* the window (evicted once
the window fills past `maxlen`) does not permanently poison the
running incremental sums -- see `_finite_sum`/`_finite_sum_sq` below.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from qml_observer.statistics.loss import loss_slope


class RollingWindow:
    """A bounded, incrementally-aggregated rolling window of scalar values.

    Not thread-safe, consistent with the rest of the core/statistics
    layers in v0.1 (addendum, Concurrency / Distributed Training).
    """

    def __init__(self, maxlen: int):
        """Create an empty rolling window.

        Args:
            maxlen: Maximum number of most-recent values retained. Must
                be a positive int.

        Raises:
            TypeError: If `maxlen` is not an int (or is a bool).
            ValueError: If `maxlen` is less than 1.
        """
        if not isinstance(maxlen, int) or isinstance(maxlen, bool):
            raise TypeError(f"maxlen must be an int, got {type(maxlen)!r}")
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")

        self._maxlen = maxlen
        self._values: deque[float] = deque(maxlen=maxlen)

        # Incremental aggregates over the *finite* entries currently in
        # the window only. Non-finite (NaN/Inf) entries are tracked by
        # count alone; mean()/variance() fall back to a direct
        # recomputation over `self._values` whenever the window
        # currently contains any non-finite entry, which keeps the
        # result numerically correct even after that entry is evicted.
        self._finite_sum = 0.0
        self._finite_sum_sq = 0.0
        self._finite_count = 0

    def append(self, value: float) -> None:
        """Append a value to the window, evicting the oldest if full.

        Args:
            value: The scalar value to record. NaN/Inf are accepted --
                they represent meaningful signal (e.g. a diverging
                metric), not invalid input.

        Raises:
            TypeError: If `value` is not a real number (or is a bool).
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"value must be a number, got {type(value)!r}")
        value = float(value)

        if len(self._values) == self._maxlen:
            evicted = self._values[0]
            if math.isfinite(evicted):
                self._finite_sum -= evicted
                self._finite_sum_sq -= evicted * evicted
                self._finite_count -= 1

        self._values.append(value)

        if math.isfinite(value):
            self._finite_sum += value
            self._finite_sum_sq += value * value
            self._finite_count += 1

    def values(self) -> list[float]:
        """Return the current window contents, oldest first."""
        return list(self._values)

    def __len__(self) -> int:
        """Number of values currently held (<= maxlen)."""
        return len(self._values)

    def _has_nonfinite(self) -> bool:
        return self._finite_count < len(self._values)

    def mean(self) -> float | None:
        """Mean of the values currently in the window.

        Returns:
            `None` if the window is empty. Otherwise the mean as a
            float -- `nan`/`inf` propagate if the window currently
            contains a non-finite value, since that is itself the
            signal the caller needs to see, not an error to hide.
        """
        n = len(self._values)
        if n == 0:
            return None
        if self._has_nonfinite():
            return float(np.mean(self._values))
        return self._finite_sum / n

    def variance(self) -> float | None:
        """Population variance (`ddof=0`) of the values in the window.

        Returns:
            `None` if the window holds fewer than 2 values (addendum
            §7: variance is undefined for 0 or 1 observations, and that
            is distinct from a computed `nan`). Otherwise the variance
            as a float; `nan`/`inf` propagate if the window currently
            contains a non-finite value.
        """
        n = len(self._values)
        if n < 2:
            return None
        if self._has_nonfinite():
            return float(np.var(self._values))
        mean = self._finite_sum / n
        var = self._finite_sum_sq / n - mean * mean
        # Guard against tiny negative results from floating-point
        # cancellation in the incremental formula (mathematically
        # variance is never negative).
        return max(var, 0.0)

    def slope(self) -> float | None:
        """Least-squares slope of the windowed values versus index.

        Values are treated as evenly spaced (index `0, 1, ..., n - 1`
        over the current window), matching `statistics.loss.loss_slope`.

        Returns:
            `None` if the window holds fewer than 2 values (undefined,
            per addendum §7). Otherwise the fitted slope as a float;
            `nan` if the window contains a non-finite value.
        """
        if len(self._values) < 2:
            return None
        return loss_slope(self._values)

    def reset(self) -> None:
        """Clear the window back to empty."""
        self._values = deque(maxlen=self._maxlen)
        self._finite_sum = 0.0
        self._finite_sum_sq = 0.0
        self._finite_count = 0
