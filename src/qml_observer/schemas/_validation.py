"""Shared validation helpers used by schema `__post_init__` hooks.

Design intent for Issue #9 ("Add schema validation"):

- Schemas should reject structurally wrong data (wrong types, impossible
  values like a negative qubit count) loudly and immediately, since that
  almost always indicates an adapter bug rather than real training
  behavior.
- Schemas must NOT reject values that are legitimate (if unhealthy)
  training signals. Per addendum §7, NaN/Inf loss or gradient values from
  a diverging optimizer are meaningful — the detector layer (Milestone 4)
  is responsible for classifying those as `IssueType.UNSTABLE`, not the
  schema layer for rejecting them outright. Helpers below therefore treat
  NaN/Inf as "not violating a non-negativity/ordering constraint" while
  still enforcing that the value is at least the right *type* of number.
- These helpers raise `TypeError`/`ValueError` directly. They are called
  from `__post_init__`, which runs synchronously at construction time —
  callers upstream (adapters, the monitor) are responsible for catching
  these per the project's fail-open policy (addendum §1); schemas
  themselves have no fail-open behavior of their own to preserve.
"""

import math


def check_type(value: object, expected: type | tuple[type, ...], name: str) -> None:
    """Raise TypeError if `value` is not an instance of `expected`."""
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be of type {expected!r}, got {type(value)!r}")


def check_non_empty_str(value: str, name: str) -> None:
    """Raise ValueError if `value` is not a non-empty string."""
    check_type(value, str, name)
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def check_non_negative_int(value: int | None, name: str) -> None:
    """Raise if `value` is not None and is a negative int (or non-int)."""
    if value is None:
        return
    check_type(value, int, name)
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise TypeError(f"{name} must be an int, got bool")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def check_finite_number(value: float | int | None, name: str) -> None:
    """Raise if `value` is not None and is not a real, finite number.

    Unlike `check_non_negative_number`, this rejects NaN/Inf too — used
    for fields where non-finite values are never meaningful (e.g.
    confidence scores, learning rates), as opposed to loss/gradient
    fields where NaN/Inf is itself the signal.
    """
    if value is None:
        return
    check_type(value, (int, float), name)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got bool")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value}")


def check_non_negative_number(value: float | int | None, name: str) -> None:
    """Raise if `value` is not None, not numeric, or negative.

    NaN is explicitly allowed through (it does not violate "non-negative"
    in any meaningful sense here, and rejecting it would break the
    fail-open handling of diverging/degenerate training signals). Inf is
    allowed for the same reason a diverging gradient norm can legitimately
    be `inf`.
    """
    if value is None:
        return
    check_type(value, (int, float), name)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got bool")
    if math.isnan(value):
        return
    if value < 0:
        raise ValueError(f"{name} must be >= 0 (or NaN), got {value}")


def check_range(value: float, lo: float, hi: float, name: str) -> None:
    """Raise if `value` is not a finite number in [lo, hi]."""
    check_type(value, (int, float), name)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got bool")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number in [{lo}, {hi}], got {value}")
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def check_str_list(value: object, name: str) -> None:
    """Raise if `value` is not a list of str."""
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value)!r}")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be a str, got {type(item)!r}")
