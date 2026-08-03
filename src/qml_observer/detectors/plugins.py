"""Third-party detector plugin API.

Milestone 14 (`future_milestones_plan.md`), Issue #103 ("Third-party
detector API"). Implements what `CONTRIBUTING.md`,
`docs/development/data_handling.md`, `docs/development/adding_detectors.md`,
and `SECURITY.md` have all promised since Milestone 10 ("a community
detector plugin API is planned for Milestone 14") -- this module *is*
that API. It does not redesign the previously-documented security
posture: plugin detectors are discovered and imported in-process, with
**no sandboxing**. A malicious or buggy plugin has full code execution in
your training process, exactly as already documented in `SECURITY.md`.
This is an accepted tradeoff for a research tool, not something this
module attempts to mitigate technically -- see this module's own
docstring section below for what modest safety nets it *does* apply
(entry-point-level fail-open discovery, a `BaseDetector` subclass check)
and what it explicitly does not (no sandboxing, no code review, no
signature verification).

## How a third-party package registers a detector

Any installed Python package can register one or more detectors under
the `qml_observer.detectors` entry-point group in its own packaging
metadata:

```toml
# a third-party package's own pyproject.toml
[project.entry-points."qml_observer.detectors"]
my_detector = "my_package.detectors:MyDetector"
```

`MyDetector` must be a `BaseDetector` subclass constructible with no
required arguments (or you can pass per-plugin constructor kwargs via
`load_detector_plugins(configs=...)`, see below).

## Discovering and using plugins

No special registration step is required to *use* a detector once you
have an instance -- `QMLMonitor(detectors=[...])` and `DiagnosisEngine`
already treat every `BaseDetector` uniformly regardless of where it came
from (see `docs/development/adding_detectors.md`). This module's job is
purely the *discovery* step for detectors installed as separate
packages, so you don't need to `import` them by name yourself:

```python
from qml_observer.detectors.plugins import load_detector_plugins
from qml_observer import QMLMonitor

plugin_detectors = load_detector_plugins()   # every installed plugin
monitor = QMLMonitor(detectors=[*builtin_detectors, *plugin_detectors])
```

Or list what's installed without instantiating anything:

```python
from qml_observer.detectors.plugins import list_detector_plugins
list_detector_plugins()   # -> {"my_detector": "my_package.detectors:MyDetector"}
```

Or via the CLI: `qml-observer plugins list`.
"""

from __future__ import annotations

import inspect
import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from qml_observer.detectors.base import BaseDetector

#: The entry-point group name third-party packages register detectors
#: under. Not versioned/namespaced further (e.g. no
#: `qml_observer.detectors.v1`) -- if this group's contract ever needs a
#: breaking change, that's itself a documented, versioned event per the
#: project's existing convention (addendum §3/Issue #108's precedent),
#: not something this constant alone should silently paper over.
DETECTOR_ENTRY_POINT_GROUP = "qml_observer.detectors"

_logger = logging.getLogger("qml_observer.detectors.plugins")


class DetectorPluginError(Exception):
    """Raised by `load_detector_plugins()` for an explicitly-requested,
    unresolvable plugin (an unknown name, or a name that fails to load).

    Discovery itself (`discover_detector_plugins()`/`list_detector_plugins()`)
    never raises this -- a single broken/incompatible plugin is skipped
    with a logged warning so the rest of the plugin ecosystem still loads
    (the same fail-open spirit as addendum §1, applied at discovery time
    rather than per training step). This exception exists only for the
    narrower case of a caller asking for a *specific* plugin by name that
    turns out not to exist or not to load.
    """


def _iter_entry_points() -> list[EntryPoint]:
    """Return every entry point registered under `DETECTOR_ENTRY_POINT_GROUP`."""
    return list(entry_points(group=DETECTOR_ENTRY_POINT_GROUP))


def list_detector_plugins() -> dict[str, str]:
    """List every registered detector plugin, without importing any of them.

    Returns:
        A mapping of entry-point name to its `module:attribute` target
        string (e.g. `{"my_detector": "my_package.detectors:MyDetector"}`),
        exactly as declared in the plugin's own package metadata. Safe to
        call even if a listed plugin's module doesn't actually exist or
        would fail to import -- nothing is imported here.
    """
    return {ep.name: ep.value for ep in _iter_entry_points()}


def discover_detector_plugins() -> dict[str, type[BaseDetector]]:
    """Import and validate every registered detector plugin.

    Each entry point is loaded (which executes that plugin's module-level
    code -- see the module docstring's no-sandboxing note) and checked to
    be a `BaseDetector` subclass. A plugin that fails to import, or that
    resolves to something other than a `BaseDetector` subclass, is
    skipped with a logged warning rather than raising, so one broken
    plugin never prevents the rest from being discovered.

    Returns:
        A mapping of entry-point name to the resolved `BaseDetector`
        subclass (not yet instantiated).
    """
    discovered: dict[str, type[BaseDetector]] = {}
    for ep in _iter_entry_points():
        try:
            obj = ep.load()
        except Exception:
            _logger.warning(
                "qml_observer: detector plugin '%s' (%s) failed to load; skipping.",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        if not (inspect.isclass(obj) and issubclass(obj, BaseDetector)):
            _logger.warning(
                "qml_observer: detector plugin '%s' (%s) does not resolve to a "
                "BaseDetector subclass (got %r); skipping.",
                ep.name,
                ep.value,
                obj,
            )
            continue

        discovered[ep.name] = obj

    return discovered


def load_detector_plugins(
    names: str | list[str] | None = None,
    *,
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[BaseDetector]:
    """Discover, instantiate, and return third-party detector plugins.

    Args:
        names: If `None` (the default), every discovered plugin is
            instantiated. If a single name or list of names is given,
            only those plugins are instantiated (in the given order) --
            useful for opting into specific community detectors rather
            than everything installed.
        configs: Optional per-plugin constructor keyword arguments, keyed
            by entry-point name (e.g. `{"my_detector": {"patience": 50}}`).
            Plugins not present in `configs` are constructed with no
            arguments.

    Returns:
        A list of instantiated `BaseDetector` objects, ready to pass to
        `QMLMonitor(detectors=[...])` or `DiagnosisEngine(detectors=[...])`
        alongside any built-in detectors.

    Raises:
        DetectorPluginError: If a name in `names` isn't a registered
            plugin, or if resolving/constructing a specifically-requested
            plugin fails. (Unlike whole-ecosystem discovery, an
            explicitly-requested plugin failing is treated as the
            caller's error, not silently skipped.)
    """
    available = discover_detector_plugins()
    configs = configs or {}

    if names is None:
        selected_names = list(available.keys())
    elif isinstance(names, str):
        selected_names = [names]
    else:
        selected_names = list(names)

    instances: list[BaseDetector] = []
    for name in selected_names:
        detector_cls = available.get(name)
        if detector_cls is None:
            raise DetectorPluginError(
                f"No detector plugin named {name!r} is registered under the "
                f"'{DETECTOR_ENTRY_POINT_GROUP}' entry-point group. "
                f"Registered plugins: {sorted(available)}"
            )
        kwargs = configs.get(name, {})
        try:
            instances.append(detector_cls(**kwargs))
        except Exception as exc:
            raise DetectorPluginError(
                f"Failed to construct detector plugin {name!r} "
                f"({detector_cls.__module__}.{detector_cls.__qualname__}) "
                f"with kwargs={kwargs!r}: {exc}"
            ) from exc

    return instances
