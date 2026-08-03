"""Milestone 14, Issue #98: basic PyTorch + QMLMonitor example.

A tiny classical `torch.nn.Module` standing in for a hybrid
quantum-classical model (e.g. one wrapping `qml.qnn.TorchLayer`): attach a
`PyTorchAdapter` to the module and optimizer, run an ordinary PyTorch
training loop, and print the diagnosis each step. Mirrors
`examples/pennylane/basic_monitor.py`'s scope -- proving the plumbing
works end to end, not tuning detectors.

Run with:
    python examples/pytorch/basic_monitor.py

Requires: pip install qml-observer[torch]
"""

from __future__ import annotations

import torch

from qml_observer import QMLMonitor
from qml_observer.adapters.pytorch.adapter import PyTorchAdapter

N_STEPS = 30


def build_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def main() -> None:
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    x = torch.randn(16, 4)
    y = torch.randn(16, 1)

    # No detectors configured, same as the PennyLane basic_monitor example:
    # this script is about the integration plumbing, not detection.
    monitor = QMLMonitor(policy="log")
    adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)

    print(f"Run ID: {monitor.run_id}\n")
    for step in range(N_STEPS):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        diagnosis = adapter.record_step(step, loss)
        print(f"step={step:>2}  loss={loss.item(): .4f}  issue={diagnosis.issue.value}")
        optimizer.step()

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    print(f"Total steps recorded: {monitor.state.step_count}")


if __name__ == "__main__":
    main()
