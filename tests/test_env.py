"""Tests for the provided environment. These should all pass out of the box.
    python -m pytest tests/test_env.py        (or)        python tests/test_env.py
"""
import os, sys
import numpy as np
import mujoco
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv
from so101_pick.env import CUBE_HALF, LIFT_HEIGHT


def set_task_pose(env, pan, height, contact=True):
    """Inject pose/contact data to test task transitions independently of control."""
    env.data.qpos[env._q_adr[0]] = pan
    env.data.qpos[env._cube_qadr + 2] = CUBE_HALF + height
    mujoco.mj_forward(env.model, env.data)
    env.data.ncon = 0  # isolate the synthetic contacts from incidental collisions
    if contact:
        for jaw in (env._fixed_jaw[0], env._moving_jaw[0]):
            c = mujoco.MjContact()
            c.geom1, c.geom2 = env._cube_geom, int(jaw)
            mujoco.mj_addContact(env.model, env.data, c)


def frozen_step(env):
    # Keep synthetic contacts intact while exercising the real step logic.
    with patch("so101_pick.env.mujoco.mj_step"):
        return env.step(np.zeros(5))


def test_reset_is_seeded():
    e = SO101PickEnv()
    o1, _ = e.reset(seed=3); o2, _ = e.reset(seed=3)
    assert np.allclose(o1, o2)


def test_curriculum_starts_near_cube_without_grasp_or_success():
    e = SO101PickEnv(randomization=0, near_cube_start_probability=1.0)
    for seed in (0, 1, 2):
        obs, info = e.reset(seed=seed)
        s = e._get_state()
        assert s["dist_ee_cube"] < 0.012
        assert 0.25 < s["gripper_opening"] < 0.35
        assert not s["is_grasped"] and not info["is_success"]
        assert not s["arm_floor_contact"]
        again, _ = e.reset(seed=seed)
        assert np.allclose(obs, again)
    e.close()


def test_disabling_curriculum_restores_standard_reset():
    standard = SO101PickEnv(randomization=0)
    curriculum = SO101PickEnv(randomization=0, near_cube_start_probability=1)
    curriculum.near_cube_start_probability = 0.0
    expected, _ = standard.reset(seed=7)
    actual, _ = curriculum.reset(seed=7)
    assert np.allclose(actual, expected)
    standard.close(); curriculum.close()


def test_interpolated_curriculum_moves_start_gradually_back_to_home():
    env = SO101PickEnv(randomization=0, near_cube_start_probability=1)
    distances = []
    for fraction in (0.0, 0.5, 1.0):
        env.set_near_cube_start_fraction(fraction)
        obs, info = env.reset(seed=0)
        state = env._get_state()
        distances.append(state["dist_ee_cube"])
        assert not state["is_grasped"] and not info["is_success"]
        assert not state["arm_floor_contact"]
    assert distances[0] < distances[1] < distances[2]
    normal = SO101PickEnv(randomization=0)
    expected, _ = normal.reset(seed=0)
    assert np.allclose(obs, expected)
    normal.close(); env.close()


def test_observation_matches_space():
    e = SO101PickEnv()
    o, _ = e.reset(seed=0)
    assert o.dtype == np.float32 and e.observation_space.contains(o)
    for _ in range(20):
        o, *_ = e.step(e.action_space.sample())
        assert o.shape == e.observation_space.shape and np.all(np.isfinite(o))


def test_zero_action_keeps_cube_still():
    e = SO101PickEnv(); e.reset(seed=0)
    p0 = e.data.xpos[e._cube_body].copy()
    for _ in range(30):
        e.step(np.zeros(5))
    assert np.linalg.norm(e.data.xpos[e._cube_body] - p0) < 0.01


def test_no_randomization_is_fixed_layout():
    e = SO101PickEnv(randomization=0.0)
    e.reset(seed=1); a = e.data.xpos[e._cube_body].copy()
    e.reset(seed=2); b = e.data.xpos[e._cube_body].copy()
    assert np.allclose(a, b, atol=1e-3)


def test_actions_are_clipped():
    e = SO101PickEnv(); e.reset(seed=0)
    e.step(np.full(5, 100.0))
    assert np.all(e._target >= e._ctrl_lo - 1e-9) and np.all(e._target <= e._ctrl_hi + 1e-9)


def test_truncation():
    e = SO101PickEnv(max_episode_steps=5); e.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, _ = e.step(np.zeros(5))
    assert trunc and not term


def test_lift_target_ends_episode_at_any_pan_angle():
    e = SO101PickEnv(randomization=0)
    for pan in (-0.4, 0.0, 0.4):
        e.reset(seed=0)
        set_task_pose(e, pan, LIFT_HEIGHT)
        obs, reward, term, trunc, info = frozen_step(e)
        assert term and not trunc and info["is_success"]
        assert reward > 0.0
        assert obs.shape == (21,)


def test_lift_requires_both_jaws_and_full_height():
    e = SO101PickEnv(randomization=0)
    for height, contact in ((0.15, True), (LIFT_HEIGHT - 0.001, True), (LIFT_HEIGHT + 0.001, False)):
        e.reset(seed=0)
        set_task_pose(e, 0.0, height, contact)
        _, _, term, _, info = frozen_step(e)
        assert not term and not info["is_success"]
    # A single jaw touching a high cube is insufficient too.
    set_task_pose(e, 0.0, LIFT_HEIGHT + 0.001, contact=False)
    c = mujoco.MjContact()
    c.geom1, c.geom2 = e._cube_geom, int(e._fixed_jaw[0])
    mujoco.mj_addContact(e.model, e.data, c)
    _, _, term, _, info = frozen_step(e)
    assert not term and not info["is_grasped"]


def test_lift_at_time_limit_counts_as_success():
    e = SO101PickEnv(randomization=0, max_episode_steps=1)
    set_task_pose(e, 0.0, LIFT_HEIGHT + 0.001)
    _, _, term, trunc, info = frozen_step(e)
    assert term and not trunc and info["is_success"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} passed")
