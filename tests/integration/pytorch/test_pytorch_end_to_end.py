"""Integration tests for the PyTorch adapter (Milestone 14, Issue #98).

Unlike `tests/unit/adapters/test_pytorch.py`, which exercises
`PyTorchAdapter` in isolation, these tests drive a full, real training
loop: a real `torch.nn.Module`, a real `torch.optim.Optimizer`, real
`loss.backward()`-computed gradients, the real detector stack, and the
real `ActionPolicy` -- end to end, exactly as `examples/pytorch/` does.
Skipped entirely if the optional `torch` dependency isn't installed.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.adapters.pytorch.adapter import PyTorchAdapter  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

PATIENCE = 15


def _detectors():
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def _healthy_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


class _FrozenModel(torch.nn.Module):
    """A model whose forward pass ignores its own (learnable-looking) parameters.

    Stands in for a genuine gradient-collapse/plateau scenario without
    needing a real barren-plateau-inducing quantum circuit: gradients
    w.r.t. `self.dead_weight` are always exactly zero since it never
    participates in the forward computation, while the *constant, poor*
    prediction keeps the loss well above `ConvergenceDetector`'s
    threshold and flat -- as opposed to a model whose output collapses to
    exactly the target (loss == 0), which the diagnosis engine correctly
    reports as genuine convergence rather than a plateau.
    """

    def __init__(self):
        super().__init__()
        self.dead_weight = torch.nn.Parameter(torch.zeros(6))

    def forward(self, x):
        constant_prediction = torch.full((x.shape[0], 1), 3.0)
        return constant_prediction + 0.0 * self.dead_weight.sum()


class TestHealthyConvergence:
    def test_does_not_falsely_stop_on_real_convergence(self):
        model = _healthy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        x = torch.randn(16, 4)
        y_true = torch.randn(16, 4) @ torch.ones(4, 1) * 0.1

        monitor = QMLMonitor(detectors=_detectors(), policy="stop")
        adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)

        stopped_early = False
        for step in range(120):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y_true)
            loss.backward()
            adapter.record_step(step, loss)
            optimizer.step()
            if monitor.should_stop():
                stopped_early = True
                break

        final = monitor.finish()
        assert stopped_early is False
        assert final.issue in (
            IssueType.HEALTHY,
            IssueType.CONVERGED,
            IssueType.INSUFFICIENT_EVIDENCE,
        )


class TestGradientCollapse:
    def test_detects_sustained_gradient_collapse(self):
        model = _FrozenModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(16, 4)

        monitor = QMLMonitor(detectors=_detectors(), policy="stop")
        adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)

        stopped_early = False
        for step in range(120):
            optimizer.zero_grad()
            loss = model(x).pow(2).mean()
            loss.backward()
            adapter.record_step(step, loss)
            optimizer.step()
            if monitor.should_stop():
                stopped_early = True
                break

        final = monitor.finish()
        assert stopped_early is True
        assert final.issue == IssueType.POSSIBLE_BARREN_PLATEAU
