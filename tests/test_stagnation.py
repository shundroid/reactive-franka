import numpy as np
import pytest

from reactive_franka.stagnation import StagnationDetector, pose_errors

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
GOAL = np.array([1.0, 0.0, 0.0])
TOLERANCES = np.array([0.02, 0.05])


def rotation(angle):
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])


def tick(
    detector,
    now,
    *,
    position=None,
    goal=None,
    quaternion=None,
    goal_quaternion=None,
    step="home",
    enabled=True,
    tolerances=TOLERANCES,
):
    return detector.update(
        now,
        step,
        np.zeros(3) if position is None else position,
        IDENTITY if quaternion is None else quaternion,
        GOAL if goal is None else goal,
        IDENTITY if goal_quaternion is None else goal_quaternion,
        tolerances,
        enabled=enabled,
    )


def test_stationary_robot_with_moving_goal_is_stalled_once_not_falsely_progressing():
    detector = StagnationDetector()
    triggered = [
        i / 30
        for i in range(241)
        if tick(
            detector,
            i / 30,
            goal=np.array([1 - i / 300, 0, 0]),
            goal_quaternion=rotation(0.5 - i / 600),
        )
    ]
    np.testing.assert_allclose(triggered, [3])


@pytest.mark.parametrize("follow_goal", [False, True])
def test_robot_progress_is_detected_even_if_moving_goal_keeps_error_constant(follow_goal):
    detector = StagnationDetector()
    for i in range(121):
        position = np.array([i / 300, 0, 0])
        goal = GOAL + position if follow_goal else GOAL
        assert not tick(detector, i / 30, position=position, goal=goal)


def test_orientation_progress_when_position_is_already_reached():
    detector = StagnationDetector()
    for i in range(121):
        assert not tick(
            detector,
            i / 30,
            goal=np.zeros(3),
            quaternion=rotation(i / 600),
            goal_quaternion=rotation(0.5),
        )


def test_rotation_inside_tolerance_cannot_mask_a_stalled_position():
    detector = StagnationDetector()
    triggered = [
        i / 30
        for i in range(121)
        if tick(detector, i / 30, quaternion=rotation(max(0, 0.03 - i / 3000)))
    ]
    np.testing.assert_allclose(triggered, [3])


def test_waits_reset_history_but_preserve_an_attempted_stall_episode():
    detector = StagnationDetector()
    for i in range(91):
        assert not tick(detector, i / 30, enabled=False)
    triggered = [i / 30 for i in range(91, 212) if tick(detector, i / 30)]
    np.testing.assert_allclose(triggered, [181 / 30])
    for i in range(212, 303):
        assert not tick(detector, i / 30, enabled=False)
    for i in range(303, 424):
        assert not tick(detector, i / 30)


def test_progress_and_task_change_allow_a_new_stall_episode():
    detector = StagnationDetector()
    triggered = []
    for i in range(601):
        position = np.array([np.clip((i / 30 - 5) * 0.1, 0, 0.4), 0, 0])
        if tick(detector, i / 30, position=position):
            triggered.append(i / 30)
    assert len(triggered) == 2
    assert triggered[0] == 3
    assert 11.8 < triggered[1] <= 12
    for i in range(601, 721):
        if tick(detector, i / 30, step="approach"):
            triggered.append(i / 30)
    assert len(triggered) == 3
    assert triggered[-1] == pytest.approx(601 / 30 + 3)


def test_quaternion_sign_is_irrelevant_and_reached_goal_does_not_stall():
    np.testing.assert_allclose(pose_errors(np.zeros(3), -IDENTITY, np.zeros(3), IDENTITY), 0)
    detector = StagnationDetector()
    for i in range(121):
        assert not tick(
            detector, i / 30, goal=np.array([0.01, 0, 0]), goal_quaternion=rotation(0.04)
        )


def test_returning_to_the_original_stalled_pose_does_not_rearm_planning():
    detector = StagnationDetector()
    triggered = []
    for i in range(451):
        now = i / 30
        recovering = 3 < now < 6
        position = (
            np.array([np.clip(-0.2 + (now - 6) * 0.2 / 3, -0.2, 0), 0, 0])
            if now > 3
            else np.zeros(3)
        )
        if tick(detector, now, position=position, enabled=not recovering):
            triggered.append(now)
    np.testing.assert_allclose(triggered, [3])
