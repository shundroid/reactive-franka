"""Convert a timed joint trajectory to cuRobo's cubic MPC seed controls."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTrajectory:
    positions: np.ndarray
    dt: float
    joint_names: list[str]

    @property
    def duration(self):
        return (len(self.positions) - 1) * self.dt

    def sample(self, times, *, initial_dt):
        """Sample the timed route with the same idle-prefix alignment as its seed."""
        moving = np.flatnonzero(np.max(np.abs(self.positions - self.positions[0]), axis=1) > 1e-5)
        offset = max(0.0, (moving[0] - 1) * self.dt - initial_dt) if len(moving) else 0.0
        plan_times = np.arange(len(self.positions)) * self.dt
        return np.array(
            [
                np.interp(np.asarray(times) + offset, plan_times, column)
                for column in self.positions.T
            ]
        )


def cubic_rollout_matrix(initial_state, knot_dt, n_knots, interpolation_steps):
    """Return position = matrix @ free_controls + fixed_start_positions.

    cuRobo pads four controls at the start from q/dq/ddq and repeats the last
    free control at the end. The first interval is fixed by the current state.
    """
    q, dq, ddq = np.asarray(initial_state, dtype=float)
    fixed = (
        q
        + np.array([-1, 0, 1, 2])[:, None] * knot_dt * dq
        + np.array([1 / 3, -1 / 6, 1 / 3, 11 / 6])[:, None] * knot_dt**2 * ddq
    )
    horizon = (n_knots + 4) * interpolation_steps + 1
    times = np.arange(horizon) / interpolation_steps
    interval = np.minimum(times.astype(int), n_knots + 3)
    u = times - interval
    basis = np.array(
        [
            (1 - u) ** 3 / 6,
            (4 - 6 * u**2 + 3 * u**3) / 6,
            (1 + 3 * u + 3 * u**2 - 3 * u**3) / 6,
            u**3 / 6,
        ]
    ).T
    matrix = np.zeros((horizon, n_knots))
    offset = np.zeros((horizon, len(q)))
    for row in range(horizon):
        for index, weight in zip(interval[row] - 4 + np.arange(4), basis[row]):
            if index < 0:
                offset[row] += weight * fixed[index + 4]
            else:
                matrix[row, min(index, n_knots - 1)] += weight
    return matrix, offset


def fit_cubic_seed(
    positions, sample_dt, initial_state, knot_dt, n_knots, interpolation_steps, *, start_time=0.0
):
    """Fit the timed plan window at start_time, without compressing its duration.

    update_seed_trajectory accepts controls, not sampled joint positions. cuRobo
    also shifts those controls left once before the first optimization. Account
    for that shift here; later ticks use cuRobo's own optimized warm start.
    """
    positions = np.asarray(positions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    if (
        positions.ndim != 2
        or len(positions) < 2
        or initial_state.shape != (3, positions.shape[1])
        or not np.isfinite(positions).all()
        or not np.isfinite(initial_state).all()
        or not np.isfinite(sample_dt)
        or sample_dt <= 0
        or not np.isfinite(knot_dt)
        or knot_dt <= 0
        or n_knots < 2
        or interpolation_steps < 1
        or not np.isfinite(start_time)
        or start_time < 0
    ):
        raise ValueError("Invalid timed trajectory or cubic MPC configuration")
    matrix, offset = cubic_rollout_matrix(initial_state, knot_dt, n_knots, interpolation_steps)
    times = np.arange(len(matrix)) * knot_dt / interpolation_steps
    samples = (
        JointTrajectory(positions, sample_dt, []).sample(times + start_time, initial_dt=knot_dt).T
    )
    shift = np.zeros((n_knots, n_knots))
    shift[np.arange(n_knots - 1), np.arange(1, n_knots)] = 1
    shift[-1, -1] = 1
    # The padded tail necessarily comes to rest. Fitting that tail to a moving
    # plan would distort the usable prefix to compensate for the forced stop.
    end = n_knots * interpolation_steps + 1
    seed = np.linalg.lstsq((matrix @ shift)[:end], (samples - offset)[:end], rcond=None)[0]
    seed[0] = initial_state[0]  # discarded by cuRobo's first shift
    return seed
