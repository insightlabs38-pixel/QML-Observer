"""Milestone 10, Issues #70-#74: webhook alerting on a generic training loop.

Demonstrates `WebhookAction` (Issue #70) delivering a structured alert
payload (Issue #71) -- optionally Slack-formatted (Issue #72), always
severity-gated against the existing `DiagnosisResult.severity` vocabulary
(Issue #73) -- to a webhook endpoint, with repeat identical alerts
suppressed while the underlying condition persists (Issue #74).

No external service is required to run this script: it spins up a tiny
local HTTP server (stdlib `http.server`, no new dependency) that just
prints whatever JSON it receives, standing in for "your alerting
backend / Slack incoming webhook / etc." This mirrors the generic
adapter's role elsewhere in the project (plan.md §9.3): the webhook
target is just an HTTP endpoint, qml-observer doesn't care what's on the
other end of it.

Run with:
    python examples/generic/webhook_alerting.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from qml_observer import QMLMonitor
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.stagnation import StagnationDetector
from qml_observer.integrations.formatters import slack_formatter
from qml_observer.integrations.webhook import WebhookAction

PATIENCE = 15
N_STEPS = 60


class _EchoAlertHandler(BaseHTTPRequestHandler):
    """Stand-in "alerting backend": accepts a POST and prints the body."""

    def do_POST(self) -> None:  # noqa: N802 (stdlib-mandated method name)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        text = body.get("text", body.get("issue", "<no text field>"))
        print(f"    [webhook received] {text}")
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args) -> None:  # noqa: A002
        pass  # keep the demo output focused on the alert content itself


def _start_local_alert_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _EchoAlertHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/alerts"


def main() -> None:
    server, url = _start_local_alert_server()
    print(f"Local alert receiver listening at {url}\n")

    monitor = QMLMonitor(
        detectors=[
            BarrenPlateauDetector(patience=PATIENCE),
            StagnationDetector(patience=PATIENCE),
        ],
        policy="warn",
        window_size=50,
    )

    # `run_id_provider`/`metrics_provider` are optional -- they let the
    # payload carry run identity and a current-metrics snapshot without
    # changing the `Action` interface; see `WebhookAction`'s docstring.
    webhook = WebhookAction(
        url,
        # This demo's "webhook" is a receiver we just spun up on localhost --
        # an intentional internal target, so we opt in explicitly (Issue
        # #75c's SSRF safeguard refuses this by default otherwise).
        allow_internal_targets=True,
        formatter=slack_formatter,
        run_id_provider=lambda: monitor.run_id,
        metrics_provider=lambda: (
            {
                "step": obs.training_event.step,
                "loss": obs.training_event.loss,
                "gradient_norm": obs.gradient.norm_l2 if obs.gradient else None,
            }
            if (obs := monitor.state.latest_observation) is not None
            else None
        ),
    )

    print(f"Run ID: {monitor.run_id}\n")
    for step in range(N_STEPS):
        # An engineered, collapsed-gradient step: constant loss, zero
        # gradient every step -- the same synthetic-plateau signature
        # `examples/pennylane/barren_plateau_demo.py` uses, here fed in
        # manually via the generic path (plan.md §9.3) instead of through
        # a real QNode.
        diagnosis = monitor.update(step=step, loss=0.71, gradients=[1e-10, -1e-10])
        result = webhook.execute(diagnosis)
        status = "sent" if result.executed else "skipped"
        print(f"step={step:>2}  issue={diagnosis.issue.value:<24}  webhook={status}")

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    server.shutdown()


if __name__ == "__main__":
    main()
