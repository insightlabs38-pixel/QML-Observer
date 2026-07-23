"""Validate detectors/DiagnosisEngine against the synthetic fixtures.

Milestone 4, Issue #32. This is the test-side proof that the synthetic
fixtures in `tests/fixtures/synthetic_runs.py` actually exercise each
detector's intended distinguishing behavior -- see plan.md §15's
benchmark categories and the addendum §3 false-positive concern.
"""

import pytest

from qml_observer import QMLMonitor
from qml_observer.detectors import (
    BarrenPlateauDetector,
    ConvergenceDetector,
    StagnationDetector,
)
from qml_observer.schemas.diagnosis import IssueType
from tests.fixtures.synthetic_runs import (
    ALL_SCENARIOS,
    artificial_plateau_run,
    convergence_run,
    healthy_learning_run,
    noise_dominated_run,
    run_through_monitor,
    stagnant_optimizer_run,
)


def _make_monitor(policy: str = "warn") -> QMLMonitor:
    return QMLMonitor(
        detectors=[
            BarrenPlateauDetector(
                gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=20
            ),
            StagnationDetector(patience=20),
            ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=20),
        ],
        policy=policy,
    )


class TestAllScenariosRegistered:
    def test_all_scenarios_produce_nonempty_step_lists(self):
        for name, generator in ALL_SCENARIOS.items():
            steps = generator(seed=0)
            assert len(steps) > 0, name


class TestHealthyLearning:
    def test_healthy_learning_is_not_flagged_as_barren_plateau(self):
        diagnosis = run_through_monitor(_make_monitor(), healthy_learning_run(seed=0))
        assert diagnosis.issue != IssueType.POSSIBLE_BARREN_PLATEAU


class TestConvergence:
    def test_convergence_run_is_diagnosed_as_converged(self):
        diagnosis = run_through_monitor(_make_monitor(), convergence_run(seed=0))
        assert diagnosis.issue == IssueType.CONVERGED
        assert diagnosis.severity == "info"

    def test_convergence_is_not_confused_with_barren_plateau(self):
        """The blueprint calls this distinction 'essential' (Volume VI-3)."""
        diagnosis = run_through_monitor(_make_monitor(), convergence_run(seed=1))
        assert diagnosis.issue != IssueType.POSSIBLE_BARREN_PLATEAU


class TestArtificialPlateau:
    def test_artificial_plateau_is_diagnosed_as_possible_barren_plateau(self):
        diagnosis = run_through_monitor(
            _make_monitor(policy="stop"), artificial_plateau_run(seed=0)
        )
        assert diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU
        assert diagnosis.severity == "critical"

    def test_artificial_plateau_triggers_should_stop(self):
        monitor = _make_monitor(policy="stop")
        run_through_monitor(monitor, artificial_plateau_run(seed=0))
        assert monitor.should_stop() is True


class TestNoiseDominated:
    def test_noise_alone_does_not_false_positive_as_barren_plateau(self):
        """High gradient variance without collapse must not read as a plateau."""
        diagnosis = run_through_monitor(_make_monitor(), noise_dominated_run(seed=0))
        assert diagnosis.issue != IssueType.POSSIBLE_BARREN_PLATEAU

    def test_noise_alone_does_not_false_positive_as_stagnation(self):
        diagnosis = run_through_monitor(_make_monitor(), noise_dominated_run(seed=0))
        assert diagnosis.issue != IssueType.STAGNATION


class TestStagnantOptimizer:
    def test_stagnant_optimizer_is_diagnosed_as_stagnation(self):
        diagnosis = run_through_monitor(_make_monitor(), stagnant_optimizer_run(seed=0))
        assert diagnosis.issue == IssueType.STAGNATION

    def test_stagnant_optimizer_is_not_confused_with_barren_plateau(self):
        """Frozen optimizer (lr=0) is a different failure mode than gradient collapse."""
        diagnosis = run_through_monitor(_make_monitor(), stagnant_optimizer_run(seed=0))
        assert diagnosis.issue != IssueType.POSSIBLE_BARREN_PLATEAU


class TestReproducibility:
    @pytest.mark.parametrize("generator_name", list(ALL_SCENARIOS))
    def test_same_seed_produces_identical_diagnosis(self, generator_name):
        generator = ALL_SCENARIOS[generator_name]
        d1 = run_through_monitor(_make_monitor(), generator(seed=42))
        d2 = run_through_monitor(_make_monitor(), generator(seed=42))
        assert d1.issue == d2.issue
        assert d1.confidence == pytest.approx(d2.confidence)
