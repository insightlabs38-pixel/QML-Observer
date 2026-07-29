"""Unit tests for qml_observer.integrations.security."""

from __future__ import annotations

import pytest

from qml_observer.integrations.security import UnsafeWebhookURLError, check_webhook_url


class TestSchemeValidation:
    def test_http_is_allowed(self):
        check_webhook_url("http://example.com/hook", allow_internal_targets=False)

    def test_https_is_allowed(self):
        check_webhook_url("https://example.com/hook", allow_internal_targets=False)

    @pytest.mark.parametrize("scheme", ["ftp", "file", "gopher", "data"])
    def test_disallowed_scheme_is_refused(self, scheme):
        with pytest.raises(UnsafeWebhookURLError, match="scheme"):
            check_webhook_url(f"{scheme}://example.com/hook", allow_internal_targets=False)


class TestInternalHostRefusal:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/hook",
            "http://localhost:8000/hook",
            "http://LOCALHOST/hook",
            "http://127.0.0.1/hook",
            "http://127.0.0.1:9000/hook",
            "http://[::1]/hook",
            "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
            "http://10.0.0.5/hook",
            "http://192.168.1.1/hook",
            "http://0.0.0.0/hook",
        ],
    )
    def test_internal_target_refused_by_default(self, url):
        with pytest.raises(UnsafeWebhookURLError, match="internal"):
            check_webhook_url(url, allow_internal_targets=False)

    @pytest.mark.parametrize(
        "url",
        ["http://localhost/hook", "http://127.0.0.1:9000/hook", "http://10.0.0.5/hook"],
    )
    def test_internal_target_allowed_when_opted_in(self, url):
        check_webhook_url(url, allow_internal_targets=True)

    def test_public_domain_is_allowed(self):
        check_webhook_url(
            "https://hooks.example.com/services/T00/B00/xyz", allow_internal_targets=False
        )

    def test_public_ip_is_allowed(self):
        # 8.8.8.8 is a public, non-private IPv4 address.
        check_webhook_url("http://8.8.8.8/hook", allow_internal_targets=False)

    def test_empty_host_refused(self):
        with pytest.raises(UnsafeWebhookURLError):
            check_webhook_url("http:///hook", allow_internal_targets=False)
