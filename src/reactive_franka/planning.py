"""Independent pose planner; main decides when to plan and how to use the result."""

import numpy as np

from .seed import JointTrajectory


class PlanningError(RuntimeError):
    """The planner could not find a route; the running MPC can continue."""


class CuroboPlanner:
    def __init__(self, robot_config, grid, device_cfg):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Scene

        self.device_cfg = device_cfg
        cfg = MotionPlannerCfg.create(
            robot=robot_config,
            scene_model=Scene(voxel=[grid]),
            device_cfg=device_cfg,
            num_trajopt_seeds=4,
            interpolation_buffer_size=4096,
            use_cuda_graph=True,
        )
        cfg.trajopt_solver_config.maximum_trajectory_dt = 1.0
        self.planner = MotionPlanner(cfg)
        voxels = self.planner.scene_collision_checker.data.voxels
        voxels.params[..., :3].copy_(device_cfg.to_device(list(grid.feature_tensor.shape)))

    def plan(self, state, position, quat_xyzw, grid):
        from curobo.types import GoalToolPose, Pose

        # This route supplies an initial guess, not commands for execution.
        # MPC anchors it to the actual q/dq/ddq. Retiming a long global plan
        # while fixing the current servo acceleration can make it infeasible.
        state = state.clone()
        state.velocity.zero_()
        state.acceleration.zero_()

        # Preserve the scene buffer address used by the planner's CUDA graphs.
        features = self.planner.scene_collision_checker.data.voxels.features
        source = grid.feature_tensor.reshape_as(features)
        if features.data_ptr() != source.data_ptr():
            features.copy_(source)
        poses = self.planner.compute_kinematics(state).tool_poses.to_dict()
        poses[self.planner.tool_frames[0]] = Pose.from_list(
            [*position, quat_xyzw[3], *quat_xyzw[:3]], device_cfg=self.device_cfg
        )
        goal = GoalToolPose.from_poses(
            poses, ordered_tool_frames=self.planner.tool_frames, num_goalset=1
        )
        result = self.planner.plan_pose(goal, state, max_attempts=5, enable_graph_attempt=1)
        if result is None or not bool(result.success.all().item()):
            raise PlanningError(f"Motion planning failed: position={position}, quat={quat_xyzw}")
        trajectory = result.get_interpolated_plan()
        if trajectory is None or trajectory.dt is None or trajectory.dt.numel() != 1:
            raise RuntimeError("Expected one interpolated trajectory with a scalar dt")
        positions = (
            trajectory.position.detach().cpu().numpy().reshape(-1, len(trajectory.joint_names))
        )
        dt = float(trajectory.dt.item())
        if len(positions) < 2 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(positions).all():
            raise RuntimeError("Invalid interpolated motion plan")
        return JointTrajectory(positions, dt, trajectory.joint_names)

    def close(self):
        self.planner.destroy()
