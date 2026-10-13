"""Run with `uv run --extra gpu --extra vision reactive-franka`."""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .control import BulletIK, CuroboMPC
from .perception import Perception
from .planning import CuroboPlanner, PlanningError
from .profiling import DISABLED_PROFILER, Profiler
from .seed import JointTrajectory
from .simulation import Simulation
from .stagnation import StagnationDetector, pose_errors
from .task import GRASP_PREDICTION_SECONDS, PickPlace
from .tracking import PoseKalmanFilter

LOG = logging.getLogger(__name__)
MAX_SEED_POSE_ERROR = np.array([0.02, 0.1])  # meters, radians at a window endpoint


@dataclass
class RecoveryPlan:
    trajectory: JointTrajectory
    step: tuple[str, float]
    started: float
    next_seed_time: float
    goal: np.ndarray | None = None
    quaternion: np.ndarray | None = None

    def stop_reason(self, current_step, now, hand, hand_quaternion):
        if current_step != self.step:
            return "task_changed"
        if now + 1e-9 < self.next_seed_time:
            return None
        error = pose_errors(hand, hand_quaternion, self.goal, self.quaternion)
        if np.any(error > MAX_SEED_POSE_ERROR):
            LOG.info("t=%.2f seed endpoint deviation=%.3fm / %.3frad", now, error[0], error[1])
            return "deviation"
        if now - self.started >= self.trajectory.duration:
            return "plan_finished"
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reactive Franka + Kalman-tracked cup + two moving obstacles"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--gui-light",
        action="store_true",
        help="Hide debug previews and disable viewport shadows/reflections for remote GUI",
    )
    parser.add_argument("--controller", choices=["curobo", "bullet-ik"], default="curobo")
    parser.add_argument("--perception", choices=["yolo", "sim-mask"], default="yolo")
    parser.add_argument("--weights", default="weights/cup-handle-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--steps", type=int, default=0, help="30 Hz control steps; 0 runs continuously"
    )
    parser.add_argument("--cycles", type=int, default=0, help="Stop after N pick/place/home cycles")
    parser.add_argument("--fast", action="store_true", help="Run without real-time pacing")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--profile",
        nargs="?",
        const="sync",
        choices=["wall", "sync"],
        help="Write stage profiles: sync waits for CUDA completion; wall preserves scheduling",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=30,
        help="Exclude the first N control steps from steady profiling statistics (default: 30)",
    )
    args = parser.parse_args()
    if (
        args.steps < 0
        or args.cycles < 0
        or args.profile_warmup_steps < 0
        or not 0 < args.confidence <= 1
    ):
        parser.error("steps/cycles/warmup must be nonnegative and confidence in (0, 1]")
    return args


def run(args):
    profiler = Profiler(args.profile)
    args.output.mkdir(parents=True, exist_ok=True)
    sim = None
    task = PickPlace()
    tracker = PoseKalmanFilter()
    updates_by_phase = {}
    steps, detections, collisions = 0, 0, 0
    started = time.monotonic()
    controller = None
    planner = None
    stagnation = StagnationDetector()
    plans = 0
    recovery = None
    seed_updates = 0
    recovery_stop = None
    last_mask = None
    interrupted = False
    error = None
    try:
        sim = Simulation(not args.headless, args.output, gui_light=args.gui_light)
        perception = Perception(args.perception, sim.cup, args.weights, args.confidence)
        frame = sim.camera.capture(sim.time)
        estimate = perception.estimate(frame)
        last_observation = estimate
        last_mask = estimate.mask if estimate is not None else None
        if args.controller == "curobo":
            controller = CuroboMPC(sim, frame, last_mask)
            planner = CuroboPlanner(controller.robot_config, controller.grid, controller.device_cfg)
        else:
            LOG.warning("Bullet IK diagnostic mode: no MPC or obstacle avoidance")
            controller = BulletIK(sim)
        with (args.output / "telemetry.jsonl").open("w") as log:
            while not args.steps or steps < args.steps:
                # step 0 reuses startup perception; do not label it as a fresh frame.
                perform_perception = steps % 3 == 0
                step_profiler = (
                    profiler if steps >= args.profile_warmup_steps else DISABLED_PROFILER
                )
                step_type = (
                    "initial" if steps == 0 else "perception" if perform_perception else "control"
                )
                with step_profiler.span("step.total@" + step_type):
                    tick_started = time.monotonic()
                    sim.update_keyboard(1 / 30)
                    # Camera + YOLO + ICP + mapping at 10 Hz; arm control and physics at 30/240 Hz.
                    if perform_perception:
                        if steps > 0:
                            with step_profiler.span("camera.total"):
                                frame = sim.camera.capture(sim.time)
                            # Fuse the fixed camera's measurements in EVERY task phase.
                            with step_profiler.span("perception.total"):
                                estimate = perception.estimate(frame, profiler=step_profiler)
                        if estimate is not None:
                            detections += 1
                            last_observation = estimate
                            last_mask = estimate.mask
                        else:
                            last_mask = None
                        previous_updates = tracker.updates
                        kalman_mode = "measurement" if estimate is not None else "prediction"
                        with step_profiler.span("kalman.total@" + kalman_mode):
                            tracked = tracker.advance(sim.time, estimate)
                        updates_by_phase[task.phase] = (
                            updates_by_phase.get(task.phase, 0) + tracker.updates - previous_updates
                        )
                        with step_profiler.span("mapping.target_exclusion"):
                            # Keep a small exclusion around the last *estimated* cup pose through
                            # brief gripper occlusion; do not fuse target fragments as obstacles.
                            if tracked is not None and (
                                tracked.is_fresh(sim.time)
                                or (
                                    task.phase == "pick"
                                    and task.step != "preapproach"
                                    and 0
                                    <= sim.time - tracked.observation_timestamp
                                    <= GRASP_PREDICTION_SECONDS
                                )
                            ):
                                valid = frame.depth > 0
                                points = frame.points(valid)
                                near_target = np.all(
                                    np.abs(points - tracked.position) < [0.08, 0.08, 0.08],
                                    axis=1,
                                )
                                region = np.zeros_like(valid)
                                region[valid] = near_target
                                last_mask = region if last_mask is None else last_mask | region
                            # During transport self-filter the attached payload using simulator labels.
                            if sim.constraint is not None:
                                last_mask = frame.body_ids == sim.cup
                        with step_profiler.span("mapping.total"):
                            controller.update_map(frame, last_mask, profiler=step_profiler)
                    else:
                        with step_profiler.span("kalman.total@prediction"):
                            tracked = tracker.advance(sim.time)
                    previous_phase = task.phase
                    with step_profiler.span("goal.total"):
                        goal, quat, grip = task.update(sim, tracked)
                    if previous_phase == "pick" and task.phase == "place":
                        Image.fromarray(sim.camera.capture(sim.time).rgb).save(
                            args.output / "handle-grasp.png"
                        )
                    step_key = (task.step, task.entered)
                    hand, hand_quat = sim.hand_pose()
                    if recovery is not None:
                        stop_reason = recovery.stop_reason(step_key, sim.time, hand, hand_quat)
                        if stop_reason is not None:
                            LOG.info(
                                "t=%.2f recovery ended: %s; continuing MPC", sim.time, stop_reason
                            )
                            recovery_stop = stop_reason
                            recovery = None
                    # Normal ticks use MPC's own warm start. Only a stalled task
                    # invokes the independent planner; the detector latches that
                    # episode so cancellation/failure cannot immediately replan.
                    stalled = stagnation.update(
                        sim.time,
                        step_key,
                        hand,
                        hand_quat,
                        goal,
                        quat,
                        task.goal_tolerances,
                        enabled=(
                            planner is not None
                            and recovery is None
                            and not task.waiting_for_target
                            and task.step not in ("close", "open")
                        ),
                    )
                    if stalled:
                        plans += 1
                        LOG.info("t=%.2f stalled %s; planning once", sim.time, task.step)
                        try:
                            with step_profiler.span("planning.total", gpu=True):
                                trajectory = planner.plan(
                                    controller.state(), goal, quat, controller.grid
                                )
                        except PlanningError as exc:
                            recovery_stop = "planning_failed"
                            LOG.warning("Recovery planning failed; continuing MPC: %s", exc)
                        else:
                            recovery = RecoveryPlan(trajectory, step_key, sim.time, sim.time)
                            recovery_stop = None
                            task.timeout = max(
                                task.timeout, sim.time - task.entered + trajectory.duration + 15.0
                            )
                            LOG.info(
                                "t=%.2f recovery plan duration=%.2fs", sim.time, trajectory.duration
                            )
                    if recovery is not None and sim.time + 1e-9 >= recovery.next_seed_time:
                        start_time = sim.time - recovery.started
                        with step_profiler.span("planning.seed", gpu=True):
                            recovery.goal, recovery.quaternion = controller.set_seed(
                                recovery.trajectory, start_time=start_time
                            )
                        recovery.next_seed_time = sim.time + controller.seed_duration
                        seed_updates += 1
                        LOG.debug(
                            "t=%.2f MPC seed for %s: plan offset=%.2fs",
                            sim.time,
                            task.step,
                            start_time,
                        )
                    sim.display_status(
                        f"Task: {task.phase} / {task.step} | Cycles: {task.cycles} | "
                        f"{args.controller} | Recovery: {'seed' if recovery is not None else 'off'} | "
                        f"Plans: {plans} | Seeds: {seed_updates}"
                    )
                    controller_stage = (
                        "mpc.total" if args.controller == "curobo" else "bullet_ik.total"
                    )
                    # Recovery follows the current plan window's endpoint pose.
                    # After cancellation, immediately resume the latest task goal.
                    mpc_goal, mpc_quat = (
                        (recovery.goal, recovery.quaternion)
                        if recovery is not None
                        else (goal, quat)
                    )
                    with step_profiler.span(controller_stage):
                        command = controller.command(mpc_goal, mpc_quat, profiler=step_profiler)
                        if not np.isfinite(command).all():
                            raise RuntimeError("Non-finite controller command")
                    with step_profiler.span("sim_step.total"):
                        sim.step(command, grip)
                        collisions += int(sim.collision_count() > 0)
                    steps += 1
                    if steps % 30 == 0:
                        record = {
                            "time": round(sim.time, 3),
                            "phase": task.phase,
                            "task_step": task.step,
                            "cycles": task.cycles,
                            "goal": goal.tolist(),
                            "goal_quaternion_xyzw": quat.tolist(),
                            "mpc_goal": mpc_goal.tolist(),
                            "mpc_goal_quaternion_xyzw": mpc_quat.tolist(),
                            "hand": sim.hand_pose()[0].tolist(),
                            "detections": detections,
                            "collision_steps": collisions,
                            "yellow_height": sim.yellow_height,
                            "mpc_failures": getattr(controller, "failures", 0),
                            "plans": plans,
                            "seed_updates": seed_updates,
                            "recovery_active": recovery is not None,
                            "recovery_stop": recovery_stop,
                            "kalman_updates": tracker.updates,
                            "observation_position": (
                                last_observation.position.tolist()
                                if last_observation is not None
                                else None
                            ),
                            "target_position": tracked.position.tolist()
                            if tracked is not None
                            else None,
                            "target_velocity": tracked.velocity.tolist()
                            if tracked is not None
                            else None,
                            "target_yaw": tracked.yaw if tracked is not None else None,
                            "target_position_std": (
                                np.sqrt(np.diag(tracked.covariance)[:3]).tolist()
                                if tracked is not None
                                else None
                            ),
                            "observation_age": (
                                sim.time - tracked.observation_timestamp
                                if tracked is not None
                                else None
                            ),
                        }
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        Image.fromarray(frame.rgb).save(args.output / "camera.png")
                    if task.waiting_for_target and sim.time > 5 and steps % 150 == 0:
                        LOG.warning(
                            "Waiting for a reliable cup mask/ICP pose; inspect outputs/camera.png"
                        )
                    if not args.fast and not (args.cycles and task.cycles >= args.cycles):
                        with step_profiler.span("pacing.sleep"):
                            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick_started)))
                if args.cycles and task.cycles >= args.cycles:
                    break
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        try:
            # Keep final screenshots outside steady statistics.
            if sim is not None:
                Image.fromarray(sim.camera.capture(sim.time).rgb).save(args.output / "camera.png")
            summary = {
                "controller": args.controller,
                "plans": plans,
                "seed_updates": seed_updates,
                "grasp_mode": "handle_top_down",
                "camera_eye": sim.camera.pose[:3, 3].tolist() if sim is not None else None,
                "perception": args.perception,
                "steps": steps,
                "sim_seconds": round(sim.time, 3) if sim is not None else 0,
                "cycles": task.cycles,
                "detections": detections,
                "kalman_updates": tracker.updates,
                "updates_by_phase": updates_by_phase,
                "collision_steps": collisions,
                "phase": task.phase,
                "wall_seconds": round(time.monotonic() - started, 2),
                "interrupted": interrupted,
                "error": error,
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            LOG.info("Run summary: %s", summary)
        finally:
            try:
                if planner is not None:
                    planner.close()
                if isinstance(controller, CuroboMPC):
                    controller.close()
                if sim is not None:
                    sim.close()
            finally:
                profiler.save(args.output / "profile.json")
    if args.cycles and task.cycles < args.cycles and not interrupted:
        raise RuntimeError(f"Only completed {task.cycles}/{args.cycles} requested cycles")
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)
    run(parse_args())


if __name__ == "__main__":
    main()
