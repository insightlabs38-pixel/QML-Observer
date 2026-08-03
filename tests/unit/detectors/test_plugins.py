"""Unit tests for qml_observer.detectors.plugins (Milestone 14, Issue #103).

Entry-point discovery is monkeypatched with lightweight fake `EntryPoint`
stand-ins (`.name`/`.value`/`.load()`) rather than real installed
packages, so these tests exercise `discover_detector_plugins()` /
`list_detector_plugins()` / `load_detector_plugins()`'s own logic --
skip-on-failure, type validation, name filtering, per-plugin config --
without needing an actual second installed distribution.
"""

from __future__ import annotations

import pytest

from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.detectors.plugins import (
    DETECTOR_ENTRY_POINT_GROUP,
    DetectorPluginError,
    discover_detector_plugins,
    list_detector_plugins,
    load_detector_plugins,
)


class _FakeEntryPoint:
    """Minimal stand-in for `importlib.metadata.EntryPoint` in tests."""

    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


class _WorkingPluginDetector(BaseDetector):
    """A minimal, valid third-party-style detector for test purposes."""

    name = "working_plugin"

    def __init__(self, patience: int = 10):
        self.patience = patience
        self._triggered = False

    def update(self, event, state) -> None:
        self._triggered = event.training_event.loss == 0.0

    def diagnose(self) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            triggered=self._triggered,
            confidence=1.0 if self._triggered else 0.0,
            evidence=["fake evidence"],
            recommendations=[],
        )

    def reset(self) -> None:
        self._triggered = False


def _not_a_detector():
    return object  # a class, but not a BaseDetector subclass


def _raises_on_load():
    raise ImportError("simulated broken plugin import")


@pytest.fixture
def patch_entry_points(monkeypatch):
    def _patch(entry_points_list):
        monkeypatch.setattr(
            "qml_observer.detectors.plugins.entry_points",
            lambda group=None: entry_points_list if group == DETECTOR_ENTRY_POINT_GROUP else [],
        )

    return _patch


class TestListDetectorPlugins:
    def test_lists_names_and_values_without_loading(self, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint("working_plugin", "pkg.mod:WorkingPluginDetector", _raises_on_load),
            ]
        )
        # Even though the loader would raise, list_detector_plugins never
        # calls .load(), so this must not raise.
        assert list_detector_plugins() == {"working_plugin": "pkg.mod:WorkingPluginDetector"}

    def test_empty_when_nothing_registered(self, patch_entry_points):
        patch_entry_points([])
        assert list_detector_plugins() == {}


class TestDiscoverDetectorPlugins:
    def test_discovers_valid_plugin(self, patch_entry_points):
        patch_entry_points(
            [_FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector)]
        )
        discovered = discover_detector_plugins()
        assert discovered == {"working_plugin": _WorkingPluginDetector}

    def test_skips_plugin_that_raises_on_load(self, patch_entry_points, caplog):
        patch_entry_points([_FakeEntryPoint("broken", "pkg.mod:Broken", _raises_on_load)])
        with caplog.at_level("WARNING"):
            discovered = discover_detector_plugins()
        assert discovered == {}
        assert "broken" in caplog.text

    def test_skips_non_detector_target(self, patch_entry_points, caplog):
        patch_entry_points([_FakeEntryPoint("notadetector", "pkg.mod:Y", _not_a_detector)])
        with caplog.at_level("WARNING"):
            discovered = discover_detector_plugins()
        assert discovered == {}
        assert "notadetector" in caplog.text

    def test_one_broken_plugin_does_not_block_others(self, patch_entry_points):
        patch_entry_points(
            [
                _FakeEntryPoint("broken", "pkg.mod:Broken", _raises_on_load),
                _FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector),
            ]
        )
        discovered = discover_detector_plugins()
        assert discovered == {"working_plugin": _WorkingPluginDetector}

    def test_no_registered_plugins_returns_empty(self, patch_entry_points):
        patch_entry_points([])
        assert discover_detector_plugins() == {}


class TestLoadDetectorPlugins:
    def test_loads_all_plugins_by_default(self, patch_entry_points):
        patch_entry_points(
            [_FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector)]
        )
        instances = load_detector_plugins()
        assert len(instances) == 1
        assert isinstance(instances[0], _WorkingPluginDetector)

    def test_loads_single_name_as_string(self, patch_entry_points):
        patch_entry_points(
            [_FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector)]
        )
        instances = load_detector_plugins("working_plugin")
        assert len(instances) == 1

    def test_unknown_name_raises(self, patch_entry_points):
        patch_entry_points([])
        with pytest.raises(DetectorPluginError, match="no_such_plugin"):
            load_detector_plugins(["no_such_plugin"])

    def test_applies_per_plugin_configs(self, patch_entry_points):
        patch_entry_points(
            [_FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector)]
        )
        (instance,) = load_detector_plugins(configs={"working_plugin": {"patience": 42}})
        assert instance.patience == 42

    def test_construction_failure_raises_plugin_error(self, patch_entry_points):
        class _AlwaysFails(BaseDetector):
            name = "always_fails"

            def __init__(self):
                raise RuntimeError("boom")

            def update(self, event, state):
                pass

            def diagnose(self):
                raise NotImplementedError

            def reset(self):
                pass

        patch_entry_points([_FakeEntryPoint("always_fails", "pkg.mod:Z", lambda: _AlwaysFails)])
        with pytest.raises(DetectorPluginError, match="Failed to construct"):
            load_detector_plugins()

    def test_loaded_plugin_participates_in_diagnosis(
        self, patch_entry_points, run_state, obs_factory
    ):
        from qml_observer.diagnosis.engine import DiagnosisEngine

        patch_entry_points(
            [_FakeEntryPoint("working_plugin", "pkg.mod:X", lambda: _WorkingPluginDetector)]
        )
        (plugin_detector,) = load_detector_plugins()
        engine = DiagnosisEngine(detectors=[plugin_detector])
        observation = obs_factory(step=0, loss=0.0)
        result = engine.evaluate(observation, run_state)
        assert result is not None
