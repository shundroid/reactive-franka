"""Detect lack of robot-caused progress, ignoring motion of the goal itself."""

from collections import deque

import numpy as np

WINDOW_SECONDS = 3.0
MIN_PROGRESS = np.array([0.005, 0.03])  # meters, radians per window


def pose_errors(position, quaternion, goal, goal_quaternion):
    cosine = abs(np.dot(quaternion, goal_quaternion)) / (
        np.linalg.norm(quaternion) * np.linalg.norm(goal_quaternion)
    )
    return np.array(
        [
            np.linalg.norm(position - goal),
            2 * np.arccos(np.clip(cosine, 0, 1)),
        ]
    )


class StagnationDetector:
    def __init__(self):
        self.step = None
        self.previous = None
        self.history = deque()
        self.attempt_error = None

    def update(
        self, now, step, position, quaternion, goal, goal_quaternion, tolerances, *, enabled=True
    ):
        if step != self.step:
            self.step = step
            self.previous = None
            self.history.clear()
            self.attempt_error = None
        if not enabled:
            self.previous = None
            self.history.clear()
            return False
        errors = pose_errors(position, quaternion, goal, goal_quaternion)
        active = errors > tolerances
        progress = np.zeros(2)
        if self.previous is not None:
            # Evaluate both poses against the SAME current goal. A stationary
            # robot has zero progress even if the target moves closer to it.
            progress = pose_errors(*self.previous, goal, goal_quaternion) - errors
        self.previous = (position.copy(), quaternion.copy())
        if not active.any():
            self.history.clear()
            self.attempt_error = None
        self.history.append((now, progress))
        while len(self.history) > 1 and self.history[1][0] <= now - WINDOW_SECONDS + 1e-9:
            self.history.popleft()
        if now - self.history[0][0] < WINDOW_SECONDS - 1e-9 or not active.any():
            return False
        progress = np.sum([value for _, value in list(self.history)[1:]], axis=0)
        if np.any(active & (progress >= MIN_PROGRESS)):
            # Recovery may move away from the task goal. Returning to the
            # original stalled pose is not progress that warrants another plan.
            if self.attempt_error is not None and np.any(
                active & (self.attempt_error - errors >= MIN_PROGRESS)
            ):
                self.attempt_error = None
            return False
        if self.attempt_error is not None:
            return False
        self.attempt_error = errors.copy()
        return True
