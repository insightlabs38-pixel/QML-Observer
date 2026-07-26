"""Unit tests for qml_observer.telemetry.consent."""

import json

import pytest

from qml_observer.telemetry import consent


@pytest.fixture
def consent_path(tmp_path):
    return tmp_path / "telemetry.json"


class TestDefaultState:
    def test_disabled_when_no_file_exists(self, consent_path):
        assert consent.is_enabled(consent_path) is False

    def test_has_not_been_asked_when_no_file_exists(self, consent_path):
        assert consent.has_been_asked(consent_path) is False

    def test_malformed_file_fails_safe_to_disabled(self, consent_path):
        consent_path.write_text("{not valid json")
        assert consent.is_enabled(consent_path) is False


class TestEnableDisable:
    def test_enable_persists_and_is_read_back(self, consent_path):
        consent.enable(consent_path)
        assert consent.is_enabled(consent_path) is True
        assert consent.has_been_asked(consent_path) is True

    def test_disable_persists(self, consent_path):
        consent.enable(consent_path)
        consent.disable(consent_path)
        assert consent.is_enabled(consent_path) is False

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "telemetry.json"
        consent.enable(nested)
        assert nested.exists()

    def test_file_contents_are_plain_json(self, consent_path):
        consent.enable(consent_path)
        data = json.loads(consent_path.read_text())
        assert data == {"enabled": True}


class TestPromptForConsent:
    def test_returns_existing_decision_without_reprompting(self, consent_path, monkeypatch):
        consent.enable(consent_path)
        # If it tried to prompt again, input() would raise (no stdin mocked).
        assert consent.prompt_for_consent(consent_path) is True

    def test_non_interactive_defaults_to_disabled_and_does_not_persist(
        self, consent_path, monkeypatch
    ):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        result = consent.prompt_for_consent(consent_path)
        assert result is False
        assert consent.has_been_asked(consent_path) is False

    def test_interactive_yes_enables_and_persists(self, consent_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "y")
        result = consent.prompt_for_consent(consent_path)
        assert result is True
        assert consent.is_enabled(consent_path) is True

    def test_interactive_default_no_disables(self, consent_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "")
        result = consent.prompt_for_consent(consent_path)
        assert result is False
        assert consent.is_enabled(consent_path) is False
