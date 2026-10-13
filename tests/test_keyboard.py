import numpy as np
import pybullet as p
import pytest

from reactive_franka.control import mapping_depth
from reactive_franka.simulation import (
    HOME,
    YELLOW_HEIGHT_LIMITS,
    YELLOW_SPEED,
    YELLOW_Y_LIMITS,
    Simulation,
)


@pytest.fixture
def sim(tmp_path):
    scene = Simulation(False, tmp_path)
    try:
        yield scene
    finally:
        scene.close()


def test_yellow_bar_starts_on_table_is_fixed_and_enters_mapping(sim):
    lower, upper = p.getAABB(sim.yellow_obstacle, physicsClientId=sim.client)
    np.testing.assert_allclose(lower, [0.26, -0.25, 0], atol=1e-8)
    np.testing.assert_allclose(upper, [0.34, 0.25, 0.06], atol=1e-8)
    assert p.getDynamicsInfo(sim.yellow_obstacle, -1, physicsClientId=sim.client)[0] == 0
    assert sim.yellow_obstacle in sim.obstacles
    frame = sim.camera.capture(0)
    mask = frame.body_ids == sim.yellow_obstacle
    assert np.count_nonzero(mask) > 100
    depth = mapping_depth(frame, sim.robot, frame.body_ids == sim.cup)
    assert np.count_nonzero(depth[mask]) > 100


def test_held_keys_move_continuously_release_stops_and_both_keys_cancel(sim, monkeypatch):
    sim.gui = True  # Exercise GUI input on a DIRECT client without a display.
    events = {p.B3G_UP_ARROW: p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED}
    monkeypatch.setattr(p, "getKeyboardEvents", lambda **kwargs: events.copy())
    sim.update_keyboard(1 / 30)
    events.clear()
    for _ in range(29):
        sim.update_keyboard(1 / 30)
    expected = YELLOW_HEIGHT_LIMITS[0] + YELLOW_SPEED
    assert sim.yellow_height == pytest.approx(expected)
    events[p.B3G_DOWN_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(1 / 30)
    assert sim.yellow_height == pytest.approx(expected)
    events.clear()
    events[p.B3G_UP_ARROW] = p.KEY_IS_DOWN | p.KEY_WAS_RELEASED
    sim.update_keyboard(1 / 30)
    assert sim.yellow_height == pytest.approx(expected - YELLOW_SPEED / 30)
    events.clear()
    events[p.B3G_DOWN_ARROW] = p.KEY_WAS_RELEASED
    sim.update_keyboard(1 / 30)
    stopped = sim.yellow_height
    events.clear()
    sim.update_keyboard(1)
    assert sim.yellow_height == stopped
    pos, quat = p.getBasePositionAndOrientation(sim.yellow_obstacle, physicsClientId=sim.client)
    np.testing.assert_allclose(pos, [0.30, 0, stopped])
    np.testing.assert_allclose(quat, [0, 0, 0, 1])


def test_horizontal_keys_share_speed_allow_diagonal_motion_and_release_independently(
    sim, monkeypatch
):
    sim.gui = True
    events = {p.B3G_RIGHT_ARROW: p.KEY_IS_DOWN, p.B3G_UP_ARROW: p.KEY_IS_DOWN}
    monkeypatch.setattr(p, "getKeyboardEvents", lambda **kwargs: events.copy())
    sim.update_keyboard(0.1)
    events.clear()
    sim.update_keyboard(0.1)
    assert sim.yellow_y == pytest.approx(0.2 * YELLOW_SPEED)
    assert sim.yellow_height == pytest.approx(0.03 + 0.2 * YELLOW_SPEED)
    events[p.B3G_LEFT_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(0.1)
    assert sim.yellow_y == pytest.approx(0.2 * YELLOW_SPEED)
    assert sim.yellow_height == pytest.approx(0.03 + 0.3 * YELLOW_SPEED)
    events.clear()
    events[p.B3G_RIGHT_ARROW] = p.KEY_WAS_RELEASED
    sim.update_keyboard(0.1)
    assert sim.yellow_y == pytest.approx(0.1 * YELLOW_SPEED)
    events.clear()
    events[p.B3G_LEFT_ARROW] = events[p.B3G_UP_ARROW] = p.KEY_WAS_RELEASED
    sim.update_keyboard(0.1)
    stopped = (sim.yellow_y, sim.yellow_height)
    events.clear()
    sim.update_keyboard(1)
    assert (sim.yellow_y, sim.yellow_height) == stopped
    position, quat = p.getBasePositionAndOrientation(
        sim.yellow_obstacle, physicsClientId=sim.client
    )
    np.testing.assert_allclose(position, [0.30, *stopped])
    np.testing.assert_allclose(quat, [0, 0, 0, 1])


def test_limits_and_automatic_motion_do_not_overwrite_yellow_height(sim, monkeypatch):
    sim.gui = True
    events = {p.B3G_UP_ARROW: p.KEY_IS_DOWN}
    monkeypatch.setattr(p, "getKeyboardEvents", lambda **kwargs: events.copy())
    sim.update_keyboard(20)
    assert sim.yellow_height == YELLOW_HEIGHT_LIMITS[1]
    events[p.B3G_UP_ARROW] = p.KEY_WAS_RELEASED
    events[p.B3G_DOWN_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(20)
    assert sim.yellow_height == YELLOW_HEIGHT_LIMITS[0]
    events[p.B3G_DOWN_ARROW] = p.KEY_WAS_RELEASED
    events[p.B3G_UP_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(1)
    events.clear()
    events[p.B3G_UP_ARROW] = p.KEY_WAS_RELEASED
    events[p.B3G_RIGHT_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(20)
    assert sim.yellow_y == YELLOW_Y_LIMITS[1]
    events[p.B3G_RIGHT_ARROW] = p.KEY_WAS_RELEASED
    events[p.B3G_LEFT_ARROW] = p.KEY_IS_DOWN
    sim.update_keyboard(20)
    assert sim.yellow_y == YELLOW_Y_LIMITS[0]
    before = p.getBasePositionAndOrientation(sim.yellow_obstacle, physicsClientId=sim.client)
    sim.move_obstacles(3)
    sim.step(HOME, 0.04)
    after = p.getBasePositionAndOrientation(sim.yellow_obstacle, physicsClientId=sim.client)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[1], before[1])
    assert p.getBasePositionAndOrientation(sim.obstacle, physicsClientId=sim.client)[0][0] != 0.52
    assert (
        p.getBasePositionAndOrientation(sim.air_obstacles[0], physicsClientId=sim.client)[0][1] != 0
    )


def test_headless_does_not_poll_keyboard(sim, monkeypatch):
    def unexpected_poll(**kwargs):
        pytest.fail("headless input should not poll GUI keys")

    monkeypatch.setattr(p, "getKeyboardEvents", unexpected_poll)
    sim.update_keyboard(1)
    assert sim.yellow_height == YELLOW_HEIGHT_LIMITS[0]


def test_robot_contacts_with_yellow_bar_are_counted(sim):
    p.resetBasePositionAndOrientation(
        sim.yellow_obstacle, sim.hand_pose()[0], [0, 0, 0, 1], physicsClientId=sim.client
    )
    p.performCollisionDetection(physicsClientId=sim.client)
    assert sim.collision_count() > 0
