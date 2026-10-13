import numpy as np
import pytest

from reactive_franka.main import RecoveryPlan
from reactive_franka.seed import JointTrajectory


def plan():
    trajectory = JointTrajectory(np.zeros((41, 7)), 0.025, [])
    return RecoveryPlan(
        trajectory, ("home", 0.0), 3.0, 3.5, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    )


def test_stop_checks_run_at_window_end_and_task_change_stops_immediately():
    recovery = plan()
    assert recovery.stop_reason(("home", 0), 3.4, np.ones(3), recovery.quaternion) is None
    assert (
        recovery.stop_reason(("pick", 3.4), 3.4, np.ones(3), recovery.quaternion) == "task_changed"
    )
    assert recovery.stop_reason(("home", 0), 3.5, np.ones(3), recovery.quaternion) == "deviation"
    assert recovery.stop_reason(("home", 0), 3.5, np.zeros(3), recovery.quaternion) is None
    assert recovery.stop_reason(("home", 0), 4, np.zeros(3), recovery.quaternion) == "plan_finished"


@pytest.mark.parametrize("angle", [0.09, 0.11])
def test_orientation_deviation_uses_the_same_cartesian_goal_as_mpc(angle):
    recovery = plan()
    quat = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
    result = recovery.stop_reason(("home", 0), 3.5, np.zeros(3), quat)
    assert result == ("deviation" if angle > 0.1 else None)


def test_endpoint_quaternion_sign_does_not_change_the_stop_condition():
    recovery = plan()
    assert recovery.stop_reason(("home", 0), 3.5, np.zeros(3), -recovery.quaternion) is None
