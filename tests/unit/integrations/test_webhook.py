"""Unit tests for qml_observer.integrations.webhook.WebhookAction."""

from __future__ import annotations

import json
import urllib.error

import pytest

from qml_observer.integrations.formatters import slack_formatter
from qml_observer.integrations.security import UnsafeWebhookURLError
from qml_observer.integrations.webhook import WebhookAction


class _FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestConstruction:
    def test_rejects_empty_url(self):
        with pytest.raises(ValueError):
            WebhookAction("")

    def test_rejects_blank_url(self):
        with pytest.raises(ValueError):
            WebhookAction("   ")

    def test_rejects_invalid_min_severity(self):
        with pytest.raises(ValueError):
            WebhookAction("https://example.com/hook", min_severity="urgent")

    def test_name(self):
        assert WebhookAction("https://example.com/hook").name == "webhook"

    def test_url_and_min_severity_properties(self):
        action = WebhookAction("https://example.com/hook", min_severity="critical")
        assert action.url == "https://example.com/hook"
        assert action.min_severity == "critical"

    def test_rejects_internal_target_by_default(self):
        with pytest.raises(UnsafeWebhookURLError):
            WebhookAction("http://localhost:9000/hook")

    def test_allows_internal_target_when_opted_in(self):
        WebhookAction("http://localhost:9000/hook", allow_internal_targets=True)

    def test_rejects_disallowed_scheme(self):
        with pytest.raises(UnsafeWebhookURLError):
            WebhookAction("ftp://example.com/hook")


class TestExecuteDelivery:
    def test_posts_default_payload(self, monkeypatch, critical_diagnosis):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        result = action.execute(critical_diagnosis)

        assert result.executed is True
        assert result.action_name == "webhook"
        assert captured["url"] == "https://example.com/hook"
        assert captured["body"]["severity"] == "critical"
        assert captured["body"]["issue"] == "possible_barren_plateau"
        assert captured["headers"]["Content-type"] == "application/json"

    def test_uses_slack_formatter_when_configured(self, monkeypatch, critical_diagnosis):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook", formatter=slack_formatter)
        action.execute(critical_diagnosis)

        assert "attachments" in captured["body"]

    def test_run_id_and_metrics_providers_are_included(self, monkeypatch, critical_diagnosis):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction(
            "https://example.com/hook",
            run_id_provider=lambda: "run-xyz",
            metrics_provider=lambda: {"step": 10, "loss": 0.2},
        )
        action.execute(critical_diagnosis)

        assert captured["body"]["run_id"] == "run-xyz"
        assert captured["body"]["current_metrics"] == {"step": 10, "loss": 0.2}

    def test_broken_provider_is_swallowed_not_raised(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        def broken_provider():
            raise RuntimeError("boom")

        action = WebhookAction("https://example.com/hook", run_id_provider=broken_provider)
        result = action.execute(critical_diagnosis)
        assert result.executed is True  # delivery still succeeds, just without run_id


class TestSeverityGating:
    def test_skips_info_severity_by_default(self, healthy_diagnosis):
        action = WebhookAction("https://example.com/hook")
        result = action.execute(healthy_diagnosis)
        assert result.executed is False
        assert "below min_severity" in result.message

    def test_min_severity_critical_skips_warning(self, monkeypatch, warning_diagnosis):
        action = WebhookAction("https://example.com/hook", min_severity="critical")
        result = action.execute(warning_diagnosis)
        assert result.executed is False


class TestDeduplication:
    def test_second_identical_alert_is_suppressed(self, monkeypatch, critical_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        first = action.execute(critical_diagnosis)
        second = action.execute(critical_diagnosis)

        assert first.executed is True
        assert second.executed is False
        assert "duplicate" in second.message.lower()
        assert len(calls) == 1

    def test_changed_severity_refires(self, monkeypatch, critical_diagnosis, warning_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        action.execute(critical_diagnosis)
        result = action.execute(warning_diagnosis)

        assert result.executed is True
        assert len(calls) == 2

    def test_deduplicate_false_always_fires(self, monkeypatch, critical_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook", deduplicate=False)
        action.execute(critical_diagnosis)
        result = action.execute(critical_diagnosis)

        assert result.executed is True
        assert len(calls) == 2

    def test_reset_clears_dedup_memory(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        action.execute(critical_diagnosis)
        action.reset()
        result = action.execute(critical_diagnosis)
        assert result.executed is True


class TestCooldown:
    def test_cooldown_none_suppresses_forever_like_issue_74(self, monkeypatch, critical_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        fake_clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: fake_clock[0])

        action = WebhookAction("https://example.com/hook")
        action.execute(critical_diagnosis)
        fake_clock[0] += 10_000
        result = action.execute(critical_diagnosis)

        assert result.executed is False
        assert len(calls) == 1

    def test_cooldown_elapsed_allows_resend(self, monkeypatch, critical_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        fake_clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: fake_clock[0])

        action = WebhookAction("https://example.com/hook", cooldown_seconds=60.0)
        first = action.execute(critical_diagnosis)
        fake_clock[0] += 30  # still within cooldown
        second = action.execute(critical_diagnosis)
        fake_clock[0] += 31  # now past the 60s cooldown (61s elapsed since first)
        third = action.execute(critical_diagnosis)

        assert first.executed is True
        assert second.executed is False
        assert "remaining" in second.message.lower()
        assert third.executed is True
        assert len(calls) == 2

    def test_cooldown_resets_the_clock_on_each_resend(self, monkeypatch, critical_diagnosis):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        fake_clock = [0.0]
        monkeypatch.setattr("time.monotonic", lambda: fake_clock[0])

        action = WebhookAction("https://example.com/hook", cooldown_seconds=60.0)
        action.execute(critical_diagnosis)
        fake_clock[0] += 61
        action.execute(critical_diagnosis)  # resend #2, resets cooldown clock
        fake_clock[0] += 30  # only 30s since resend #2
        third = action.execute(critical_diagnosis)

        assert third.executed is False
        assert len(calls) == 2


class TestRedactEvidence:
    def test_strips_evidence_and_metrics_from_body(self, monkeypatch, critical_diagnosis):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction(
            "https://example.com/hook",
            redact_evidence=True,
            metrics_provider=lambda: {"step": 1},
        )
        action.execute(critical_diagnosis)

        assert captured["body"]["evidence"] == []
        assert captured["body"]["current_metrics"] == {}
        assert captured["body"]["redacted"] is True

    def test_default_does_not_redact(self, monkeypatch, critical_diagnosis):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(200)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        action.execute(critical_diagnosis)

        assert captured["body"]["evidence"] != []
        assert captured["body"]["redacted"] is False


class TestFailureHandling:
    def test_http_error_is_reported_not_raised(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 500, "Internal Error", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        result = action.execute(critical_diagnosis)
        assert result.executed is False
        assert "500" in result.message

    def test_connection_error_is_reported_not_raised(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        result = action.execute(critical_diagnosis)
        assert result.executed is False
        assert "connection error" in result.message.lower()

    def test_non_2xx_status_is_reported_not_raised(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            return _FakeResponse(404)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook")
        result = action.execute(critical_diagnosis)
        assert result.executed is False
        assert "404" in result.message

    def test_broken_formatter_is_reported_not_raised(self, monkeypatch, critical_diagnosis):
        def broken_formatter(payload):
            raise RuntimeError("formatter exploded")

        action = WebhookAction("https://example.com/hook", formatter=broken_formatter)
        result = action.execute(critical_diagnosis)
        assert result.executed is False
        assert "formatter" in result.message.lower()

    def test_timeout_is_reported_not_raised(self, monkeypatch, critical_diagnosis):
        def fake_urlopen(request, timeout=None):
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        action = WebhookAction("https://example.com/hook", timeout=1.0)
        result = action.execute(critical_diagnosis)
        assert result.executed is False
        assert "timed out" in result.message.lower()
