import os
from types import SimpleNamespace

import numpy as np
import pytest
from test_task import Sim, observation

from reactive_franka.seed import JointTrajectory, cubic_rollout_matrix, fit_cubic_seed
from reactive_franka.task import PickPlace


def test_transition_returns_current_step_goal_immediately():
    task, sim = PickPlace(), Sim()
    goal, _, _ = task.update(sim, observation())
    sim.hand = goal.copy()
    sim.time = 0.1
    new_goal, _, _ = task.update(sim, observation(sim.time))
    assert task.step == "approach"
    assert task.entered == sim.time
    assert new_goal[2] < goal[2]
    sim.time += 1 / 30
    np.testing.assert_allclose(task.update(sim, observation(sim.time))[0], new_goal)


def test_long_planned_step_has_budget_but_next_step_resets_it():
    task, sim = PickPlace(), Sim()
    task.update(sim, observation())
    task.transition("home", 0)
    task.timeout = 43
    sim.hand += [0.1, 0.0, 0.0]
    sim.time = 28
    task.update(sim, observation(sim.time))
    assert task.step == "home"
    sim.hand, sim.quat = task.home.copy(), task.home_quaternion.copy()
    task.update(sim, observation(sim.time))
    assert task.step == "preapproach"
    assert task.timeout == 25


def test_seed_preserves_physical_time_instead_of_squeezing_whole_plan():
    q = np.linspace(-0.5, 0.5, 7)
    velocity = np.full(7, 0.1)
    initial = np.array([q, velocity, np.zeros(7)])
    positions = q + np.arange(1001)[:, None] * 0.025 * velocity
    seed = fit_cubic_seed(positions, 0.025, initial, 1 / 30, 16, 4)
    shifted = np.concatenate([seed[1:], seed[-1:]])
    matrix, offset = cubic_rollout_matrix(initial, 1 / 30, 16, 4)
    rollout = matrix @ shifted + offset
    times = np.arange(len(rollout)) / 120
    # Endpoint padding changes the last intervals, but not the inner prefix.
    np.testing.assert_allclose(rollout[:65], q + times[:65, None] * velocity, atol=1e-10)
    assert np.max(np.abs(rollout[-1] - positions[-1])) > 2
    assert seed.shape == (16, 7)


def test_constant_plan_seed_and_invalid_inputs():
    q = np.linspace(-1, 1, 7)
    initial = np.array([q, np.zeros(7), np.zeros(7)])
    seed = fit_cubic_seed(np.tile(q, (2, 1)), 0.025, initial, 1 / 30, 16, 4)
    np.testing.assert_allclose(seed, np.tile(q, (16, 1)), atol=1e-12)
    for dt in (0, -1, np.nan):
        with pytest.raises(ValueError):
            fit_cubic_seed(np.tile(q, (2, 1)), dt, initial, 1 / 30, 16, 4)
    for start_time in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            fit_cubic_seed(np.tile(q, (2, 1)), 0.025, initial, 1 / 30, 16, 4, start_time=start_time)


def test_successive_seeds_use_later_plan_windows_and_pad_only_at_the_end():
    q = np.linspace(-0.5, 0.5, 7)
    velocity = np.full(7, 0.1)
    positions = q + np.arange(121)[:, None] * 0.025 * velocity
    for start_time in (0, 16 / 30, 32 / 30, 48 / 30):
        initial = np.array([q + start_time * velocity, velocity, np.zeros(7)])
        seed = fit_cubic_seed(positions, 0.025, initial, 1 / 30, 16, 4, start_time=start_time)
        matrix, offset = cubic_rollout_matrix(initial, 1 / 30, 16, 4)
        rollout = matrix @ np.concatenate([seed[1:], seed[-1:]]) + offset
        times = np.arange(65) / 120 + start_time
        np.testing.assert_allclose(rollout[:65], q + times[:, None] * velocity, atol=1e-10)
    initial = np.array([positions[-1], np.zeros(7), np.zeros(7)])
    seed = fit_cubic_seed(positions, 0.025, initial, 1 / 30, 16, 4, start_time=4)
    np.testing.assert_allclose(seed, np.tile(positions[-1], (16, 1)), atol=1e-12)


@pytest.mark.parametrize(
    "case", ["normal", "stalled", "step_changed", "deviation", "plan_failed", "plan_finished"]
)
def test_main_only_plans_on_stall_and_stops_recovery_without_replanning(
    tmp_path, monkeypatch, case
):
    from reactive_franka import main
    from reactive_franka.planning import PlanningError

    plan_calls, seed_calls, commands = [], [], []
    task = SimpleNamespace(
        phase="go_home",
        step="home",
        entered=0.0,
        waiting_for_target=False,
        cycles=0,
        timeout=25,
        goal_tolerances=np.array([0.02, 0.05]),
    )

    def update(sim, tracked):
        if case == "step_changed" and round(sim.time * 30) >= 110:
            task.step = "retreat"
            task.entered = 110 / 30
        goal = np.zeros(3) if case == "normal" else np.array([0.2, 0, 0])
        return goal, np.array([1, 0, 0, 0]), 0.04

    task.update = update
    monkeypatch.setattr(main, "PickPlace", lambda: task)
    frame = SimpleNamespace(rgb=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=0)

    class Scene:
        def __init__(self, *args, **kwargs):
            self.time, self.cup, self.robot, self.constraint = 0.0, 1, 0, None
            self.camera = SimpleNamespace(pose=np.eye(4), capture=lambda _: frame)
            self.hand = np.zeros(3)
            self.quat = np.array([1, 0, 0, 0])

        def step(self, *args):
            self.time += 1 / 30

        def collision_count(self):
            return 0

        def display_status(self, text):
            pass

        def hand_pose(self):
            drift = (
                np.array([0.1, 0, 0])
                if case == "deviation" and round(self.time * 30) >= 110
                else np.zeros(3)
            )
            return self.hand + drift, self.quat.copy()

        def close(self):
            pass

    class Controller:
        seed_duration = 16 / 30
        robot_config, grid, device_cfg = {}, None, None

        def __init__(self, sim, *args):
            self.sim = sim

        def state(self):
            return None

        def update_map(self, *args, **kwargs):
            pass

        def set_seed(self, trajectory, *, start_time):
            seed_calls.append((self.sim.time, start_time))
            return np.array([0.05 + 0.01 * start_time, 0, 0]), np.array([0, 1, 0, 0])

        def command(self, position, quaternion, **kwargs):
            commands.append((self.sim.time, position.copy(), quaternion.copy()))
            if np.allclose(quaternion, [0, 1, 0, 0]):
                self.sim.hand = position.copy()
                self.sim.quat = quaternion.copy()
            return np.zeros(7)

        def close(self):
            pass

    def plan(*args):
        plan_calls.append(task.step)
        if case == "plan_failed":
            raise PlanningError("No route")
        return JointTrajectory(np.zeros((41 if case == "plan_finished" else 101, 7)), 0.025, [])

    monkeypatch.setattr(main, "Simulation", Scene)
    monkeypatch.setattr(main, "CuroboMPC", Controller)
    monkeypatch.setattr(
        main, "CuroboPlanner", lambda *args: SimpleNamespace(plan=plan, close=lambda: None)
    )
    monkeypatch.setattr(
        main, "Perception", lambda *args: SimpleNamespace(estimate=lambda *a, **kw: None)
    )
    args = SimpleNamespace(
        profile=None,
        output=tmp_path,
        headless=True,
        gui_light=False,
        perception="yolo",
        weights="",
        confidence=0.25,
        controller="curobo",
        steps=140,
        profile_warmup_steps=30,
        fast=True,
        cycles=0,
    )
    summary = main.run(args)
    if case == "normal":
        assert plan_calls == []
        assert seed_calls == []
    elif case == "plan_failed":
        assert plan_calls == ["home"]
        assert seed_calls == []
    elif case == "stalled":
        assert plan_calls == ["home"]
        np.testing.assert_allclose(
            seed_calls,
            [(3, 0), (3 + 16 / 30, 16 / 30), (3 + 32 / 30, 32 / 30), (3 + 48 / 30, 48 / 30)],
        )
    else:
        assert plan_calls == ["home"]
        np.testing.assert_allclose(seed_calls, [(3, 0), (3 + 16 / 30, 16 / 30)])
    assert summary["plans"] == len(plan_calls)
    assert summary["seed_updates"] == len(seed_calls)
    for now, position, quaternion in commands:
        recovery = case == "stalled" and now >= 3 - 1e-9
        recovery |= case == "step_changed" and 3 - 1e-9 <= now < 110 / 30 - 1e-9
        recovery |= case in ("deviation", "plan_finished") and 3 - 1e-9 <= now < 3 + 32 / 30 - 1e-9
        if recovery:
            offset = [offset for time, offset in seed_calls if time <= now + 1e-9][-1]
            np.testing.assert_allclose(position, [0.05 + 0.01 * offset, 0, 0])
            np.testing.assert_allclose(quaternion, [0, 1, 0, 0])
        else:
            np.testing.assert_allclose(position, np.zeros(3) if case == "normal" else [0.2, 0, 0])
            np.testing.assert_allclose(quaternion, [1, 0, 0, 0])


def test_leading_stationary_padding_does_not_fill_the_entire_mpc_seed():
    times = np.arange(1001) * 0.025
    positions = np.maximum(0, times - 0.8)[:, None] * np.full((1, 7), 0.1)
    initial = np.zeros((3, 7))
    seed = fit_cubic_seed(positions, 0.025, initial, 1 / 30, 16, 4)
    shifted = np.concatenate([seed[1:], seed[-1:]])
    matrix, offset = cubic_rollout_matrix(initial, 1 / 30, 16, 4)
    rollout = matrix @ shifted + offset
    np.testing.assert_allclose(rollout[:5], 0, atol=1e-6)
    # Away from the boundary transition, the fitted speed stays within 1%.
    # Removing the idle prefix must not compress the moving part's duration.
    np.testing.assert_allclose(rollout[50] - rollout[38], 0.01, rtol=0.01, atol=1e-7)
    assert np.max(np.abs(rollout[-1])) < 0.07


def test_seed_route_planning_does_not_mutate_measured_velocity_or_acceleration():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    from curobo.types import DeviceCfg, JointState, Pose

    from reactive_franka.planning import CuroboPlanner

    device = DeviceCfg(device="cpu")
    names = [f"panda_joint{i}" for i in range(1, 8)]
    state = JointState.from_position(torch.zeros((1, 7)), joint_names=names)
    state.velocity.fill_(0.12)
    state.acceleration.fill_(3)
    features = torch.zeros((2, 2, 2))
    trajectory = JointState.from_position(torch.zeros((2, 7)), joint_names=names)
    trajectory.dt = torch.tensor([0.025])
    goal_position = np.array([0.45, -0.23, 0.65])

    def plan_pose(goal, start, **kwargs):
        torch.testing.assert_close(start.position, state.position)
        assert torch.count_nonzero(start.velocity) == 0
        assert torch.count_nonzero(start.acceleration) == 0
        np.testing.assert_allclose(goal.position.numpy().reshape(3), goal_position)
        return SimpleNamespace(
            success=torch.tensor([True]), get_interpolated_plan=lambda: trajectory
        )

    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.device_cfg = device
    planner.planner = SimpleNamespace(
        scene_collision_checker=SimpleNamespace(
            data=SimpleNamespace(voxels=SimpleNamespace(features=features))
        ),
        tool_frames=["panda_hand"],
        plan_pose=plan_pose,
        compute_kinematics=lambda state: SimpleNamespace(
            tool_poses=SimpleNamespace(
                to_dict=lambda: {
                    "panda_hand": Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=device)
                }
            )
        ),
    )
    planner.plan(
        state, goal_position, np.array([1, 0, 0, 0]), SimpleNamespace(feature_tensor=features)
    )
    torch.testing.assert_close(state.velocity, torch.full_like(state.velocity, 0.12))
    torch.testing.assert_close(state.acceleration, torch.full_like(state.acceleration, 3))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="opt-in CUDA test")
def test_seed_matches_cuda_rollout_and_tracks_the_plan_endpoint_pose(tmp_path, monkeypatch):
    import torch
    from curobo.types import JointState

    from reactive_franka.control import CuroboMPC
    from reactive_franka.perception import Perception
    from reactive_franka.planning import CuroboPlanner
    from reactive_franka.simulation import Simulation

    sim = Simulation(False, tmp_path)
    controller = None
    planner = None
    try:
        frame = sim.camera.capture(0)
        estimate = Perception("sim-mask", sim.cup, "").estimate(frame)
        controller = CuroboMPC(sim, frame, estimate.mask)

        def forbidden_reset(*args, **kwargs):
            raise AssertionError("Replacing an MPC seed must not reset the controller")

        monkeypatch.setattr(controller.mpc, "reset_robot", forbidden_reset)
        state = controller.state()
        transition = controller.mpc.transition_model
        q = state.position.cpu().numpy()[0]
        positions = q + np.arange(1001)[:, None] * 0.025 * np.full(7, 0.01)
        trajectory = JointTrajectory(positions, 0.025, controller.mpc.joint_names)
        controller.set_seed(trajectory)
        expected = controller.mpc.trajectory_execution_manager.get_shifteaction_dim_buffer().clone()
        initial = np.array([q, np.zeros(7), np.zeros(7)])
        matrix, offset = cubic_rollout_matrix(
            initial, transition.dt, transition.n_knots, transition.interpolation_steps
        )

        # Exercise cuRobo's actual CUDA B-spline kernel, including the padded endpoints.
        state.dt = controller.tensor([transition.dt])
        state.jerk = torch.zeros_like(state.position)
        output = JointState.zeros((1, transition.config.horizon, 7), controller.device_cfg)
        output = transition.robot_cmd_tensor_step(
            state,
            expected,
            output,
            state_idx=torch.zeros(1, device="cuda:0", dtype=torch.int32),
            use_implicit_goal_state=torch.zeros(1, device="cuda:0", dtype=torch.uint8),
        )
        np.testing.assert_allclose(
            output.position.cpu().numpy()[0],
            matrix @ expected.cpu().numpy()[0] + offset,
            atol=2e-6,
        )
        # The independent planner can return passive finger joints as well.
        # Feed its real pose plan through the same seed API before optimizing.
        planner = CuroboPlanner(controller.robot_config, controller.grid, controller.device_cfg)
        target, quat = sim.hand_pose()
        target[2] += 0.02
        plan = planner.plan(controller.state(), target, quat, controller.grid)
        assert plan.positions.shape[1] == len(plan.joint_names)
        plan_goal, plan_quat = controller.set_seed(plan)
        order = [plan.joint_names.index(name) for name in controller.mpc.joint_names]
        endpoint = plan.sample(controller.seed_duration, initial_dt=transition.dt)[order][
            controller.indices
        ]
        # Verify the planned joint endpoint is converted to the correct hand
        # pose, not the final task pose. Bullet provides an independent FK check.
        import pybullet as p

        saved = p.saveState(physicsClientId=sim.client)
        try:
            for joint, value in zip(sim.arm, endpoint):
                p.resetJointState(sim.robot, joint, value, physicsClientId=sim.client)
            expected_position, expected_quat = sim.hand_pose()
            np.testing.assert_allclose(plan_goal, expected_position, atol=1e-5)
            assert abs(np.dot(plan_quat, expected_quat)) > 1 - 1e-6
        finally:
            p.restoreState(saved, physicsClientId=sim.client)
            p.removeState(saved, physicsClientId=sim.client)
        expected = controller.mpc.trajectory_execution_manager.get_shifteaction_dim_buffer().clone()
        optimized_seeds = []
        original_optimize = controller.mpc.optimizer.optimize

        def optimize(seed, *args, **kwargs):
            optimized_seeds.append(seed.clone())
            return original_optimize(seed, *args, **kwargs)

        monkeypatch.setattr(controller.mpc.optimizer, "optimize", optimize)
        goals = []
        original_goal = controller.mpc.update_goal_tool_poses

        def update_goal(goal, **kwargs):
            goals.append(goal)
            return original_goal(goal, **kwargs)

        monkeypatch.setattr(controller.mpc, "update_goal_tool_poses", update_goal)

        def forbidden_joint_goal(*args, **kwargs):
            raise AssertionError("Recovery MPC must track a Cartesian pose goal")

        monkeypatch.setattr(controller.mpc, "update_goal_state", forbidden_joint_goal)
        command = controller.command(plan_goal, plan_quat)
        torch.testing.assert_close(optimized_seeds[0], expected)
        np.testing.assert_allclose(goals[0].position.cpu().numpy().reshape(3), plan_goal, atol=1e-6)
        np.testing.assert_allclose(
            goals[0].quaternion.cpu().numpy().reshape(4), plan_quat[[3, 0, 1, 2]], atol=1e-6
        )
        assert np.isfinite(command).all()
        assert controller.mpc._mpc_warm_start_available
        controller.set_seed(plan, start_time=controller.seed_duration)
        assert controller.mpc._mpc_warm_start_available
        actual = controller.mpc.trajectory_execution_manager.get_shifteaction_dim_buffer()
        state = controller.state()
        initial = (
            torch.stack([state.position, state.velocity, state.acceleration])
            .cpu()
            .numpy()
            .reshape(3, -1)
        )
        order = [plan.joint_names.index(name) for name in controller.mpc.joint_names]
        expected_seed = fit_cubic_seed(
            plan.positions[:, order],
            plan.dt,
            initial,
            transition.dt,
            transition.n_knots,
            transition.interpolation_steps,
            start_time=controller.seed_duration,
        )
        shifted = np.concatenate([expected_seed[1:], expected_seed[-1:]])
        np.testing.assert_allclose(actual.cpu().numpy()[0], shifted, atol=2e-6)
    finally:
        if planner is not None:
            planner.close()
        if controller is not None:
            controller.close()
        sim.close()
