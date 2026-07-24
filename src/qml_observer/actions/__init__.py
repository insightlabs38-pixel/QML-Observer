"""qml_observer.actions: the action layer (Milestone 5, Volume XI).

Public re-exports for `qml_observer.actions.*` -- turns a `DiagnosisResult`
into a concrete response (log, alert, stop), and `ActionPolicy`, which
decides which of those should run.
"""

from __future__ import annotations

from qml_observer.actions.alert import AlertAction
from qml_observer.actions.base import Action, ActionResult
from qml_observer.actions.log import LogAction
from qml_observer.actions.policies import ActionPolicy
from qml_observer.actions.stop import StopAction

__all__ = [
    "Action",
    "ActionResult",
    "ActionPolicy",
    "LogAction",
    "AlertAction",
    "StopAction",
]
