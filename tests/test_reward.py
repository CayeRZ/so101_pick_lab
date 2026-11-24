"""Self-check for TODO 1 (your reward). These tests use synthetic states, no training needed.
They FAIL with the starter reward and should PASS once your staged reward is sensible.
    python -m pytest tests/test_reward.py -v      (or)      python tests/test_reward.py
Passing is necessary, not sufficient: a good reward must still *train*.
"""
import copy, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from so101_pick import SO101PickEnv
from so101_pick.rewards import compute_reward, OVERSHOOT_HEIGHT
from so101_pick.env import LIFT_HEIGHT

ZERO = np.zeros(5, dtype=np.float32)


def base_state():
    env = SO101PickEnv(); env.reset(seed=0)
    return env._get_state()


def with_dist(s, dist):
    """Place the jaws `dist` metres straight above the cube."""
    s = copy.deepcopy(s)
    s["ee_pos"] = s["cube_pos"] + np.array([0, 0, dist], dtype=np.float32)
    s["ee_to_cube"] = s["cube_pos"] - s["ee_pos"]
    s["dist_ee_cube"] = float(dist)
    return s


def grasped(s, height):
    s = with_dist(s, 0.005)
    s.update(fixed_jaw_contact=True, moving_jaw_contact=True, is_grasped=True,
             cube_height=float(height), gripper_opening=0.05)
    s["cube_pos"] = s["cube_pos"].copy(); s["cube_pos"][2] = 0.0125 + height
    s["success"] = height >= LIFT_HEIGHT
    return s


def test_reward_is_finite_float():
    r = compute_reward(base_state(), ZERO)
    assert np.isfinite(r)


def test_getting_closer_is_rewarded():
    s = base_state()
    assert compute_reward(with_dist(s, 0.02), ZERO) > compute_reward(with_dist(s, 0.10), ZERO)


def test_grasping_beats_hovering():
    s = base_state()
    assert compute_reward(grasped(s, 0.0), ZERO) > compute_reward(with_dist(s, 0.005), ZERO)


def test_lifting_while_grasped_is_rewarded():
    s = base_state()
    assert compute_reward(grasped(s, 0.03), ZERO) > compute_reward(grasped(s, 0.0), ZERO)


def test_success_is_the_best_outcome():
    s = base_state()
    best_non_success = compute_reward(grasped(s, 0.9 * LIFT_HEIGHT), ZERO)
    assert compute_reward(grasped(s, LIFT_HEIGHT), ZERO) > best_non_success


def test_closing_gripper_in_empty_air_is_not_rewarded():
    s = with_dist(base_state(), 0.10)
    closed = copy.deepcopy(s); closed["gripper_opening"] = -0.1
    opened = copy.deepcopy(s); opened["gripper_opening"] = 0.6
    assert compute_reward(closed, ZERO) <= compute_reward(opened, ZERO) + 1e-6


def test_lifting_without_grasp_is_not_rewarded():
    """Cube 3 cm up but NOT grasped (e.g. thrown/balanced) must not look like progress."""
    s = base_state()
    g = grasped(s, 0.03)
    air = copy.deepcopy(g); air.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False)
    assert compute_reward(air, ZERO) < compute_reward(g, ZERO)


def test_stays_open_during_approach():
    for distance in (0.10, 0.05, 0.03):
        opened = with_dist(base_state(), distance)
        opened["gripper_opening"] = 0.6
        for opening in (-0.1, 0.05):
            closed = copy.deepcopy(opened)
            closed["gripper_opening"] = opening
            assert compute_reward(opened, ZERO) > compute_reward(closed, ZERO)


def test_closing_is_encouraged_when_aligned_at_cube():
    opened = with_dist(base_state(), 0.005)
    opened["gripper_opening"] = 0.6
    closed = copy.deepcopy(opened)
    closed["gripper_opening"] = 0.05
    assert compute_reward(closed, ZERO) > compute_reward(opened, ZERO)
    contact = copy.deepcopy(closed)
    contact["fixed_jaw_contact"] = True
    assert compute_reward(contact, ZERO) > compute_reward(closed, ZERO)
    assert compute_reward(grasped(opened, 0.0), ZERO) > compute_reward(contact, ZERO)


def test_excessive_opening_is_worse_than_capture_opening():
    # Regression: the observed policy stayed at ~1.6 rad without grasping.
    for distance in (0.10, 0.02, 0.005):
        s = with_dist(base_state(), distance)
        s["gripper_opening"] = 0.6
        wide = copy.deepcopy(s)
        wide["gripper_opening"] = 1.6
        assert compute_reward(s, ZERO) > compute_reward(wide, ZERO)


def test_closing_from_wide_open_has_a_meaningful_signal():
    s = with_dist(base_state(), 0.005)
    rewards = []
    for opening in (1.6, 1.2, 0.8, 0.4, 0.05):
        s["gripper_opening"] = opening
        rewards.append(compute_reward(s, ZERO))
    assert all(b - a > 0.01 for a, b in zip(rewards, rewards[1:]))


def test_lateral_misalignment_does_not_encourage_closing():
    opened = with_dist(base_state(), 0.015)
    opened["ee_to_cube"] = np.array([0.015, 0.0, 0.0])
    opened["ee_pos"] = opened["cube_pos"] - opened["ee_to_cube"]
    opened["gripper_opening"] = 0.6
    closed = copy.deepcopy(opened)
    closed["gripper_opening"] = 0.05
    assert compute_reward(opened, ZERO) > compute_reward(closed, ZERO)


def test_descending_beats_hovering_with_open_jaws():
    s = base_state()
    for opening in (0.3, 0.6):
        hover = with_dist(s, 0.02)
        lower = with_dist(s, 0.005)
        hover["gripper_opening"] = lower["gripper_opening"] = opening
        assert compute_reward(lower, ZERO) > compute_reward(hover, ZERO)


def test_dropping_or_lowering_cube_is_worse():
    held = grasped(base_state(), LIFT_HEIGHT - 0.001)
    low = copy.deepcopy(held)
    low["cube_height"] = 0.03
    dropped = copy.deepcopy(held)
    dropped.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False)
    assert compute_reward(held, ZERO) > compute_reward(low, ZERO)
    assert compute_reward(held, ZERO) > compute_reward(dropped, ZERO)


def test_height_has_no_effect_without_grasp():
    s = with_dist(base_state(), 0.005)
    airborne = copy.deepcopy(s)
    airborne["cube_height"] = LIFT_HEIGHT
    assert compute_reward(airborne, ZERO) == compute_reward(s, ZERO)


def test_waiting_below_success_height_costs_reward():
    s = grasped(base_state(), LIFT_HEIGHT - 0.001)
    s["success"] = False
    assert compute_reward(s, ZERO) < 0.0
    done = copy.deepcopy(s)
    done.update(cube_height=LIFT_HEIGHT, success=True)
    assert compute_reward(done, ZERO) > 0.0


def test_earlier_success_is_better():
    early = grasped(base_state(), LIFT_HEIGHT)
    early["step"] = 30
    late = copy.deepcopy(early)
    late["step"] = 150
    assert compute_reward(early, ZERO) > compute_reward(late, ZERO)


def test_rising_with_a_grasp_beats_holding_and_lowering_at_the_same_height():
    held = grasped(base_state(), 0.08)
    held.update(previous_cube_height=0.08, previous_is_grasped=True)
    rising = copy.deepcopy(held)
    rising["previous_cube_height"] = 0.07
    falling = copy.deepcopy(held)
    falling["previous_cube_height"] = 0.09
    assert compute_reward(rising, ZERO) > compute_reward(held, ZERO)
    assert compute_reward(held, ZERO) > compute_reward(falling, ZERO)


def test_ungrasped_upward_motion_earns_no_height_bonus():
    air = with_dist(base_state(), 0.005)
    air.update(cube_height=0.08, previous_cube_height=0.0, previous_is_grasped=False)
    stationary = copy.deepcopy(air)
    stationary["previous_cube_height"] = 0.08
    assert compute_reward(air, ZERO) == compute_reward(stationary, ZERO)


def test_losing_grasp_removes_height_credit_even_before_the_cube_falls():
    held = grasped(base_state(), 0.10)
    held.update(previous_cube_height=0.10, previous_is_grasped=True)
    released = copy.deepcopy(held)
    released.update(is_grasped=False, fixed_jaw_contact=False, moving_jaw_contact=False,
                    success=False)
    never_held = copy.deepcopy(released)
    never_held["previous_is_grasped"] = False
    assert compute_reward(released, ZERO) < compute_reward(never_held, ZERO)


def test_penalty_starts_only_above_15_point_5_cm_and_grows_with_excess():
    s = base_state()
    rewards = [compute_reward(grasped(s, height), ZERO)
               for height in (LIFT_HEIGHT, OVERSHOOT_HEIGHT, 0.156, 0.160, 0.170)]
    assert np.isclose(rewards[0], rewards[1])
    assert all(before > after for before, after in zip(rewards[1:], rewards[2:]))


def test_terminal_crossing_prefers_goal_band_over_overshoot():
    s = base_state()
    target = grasped(s, LIFT_HEIGHT)
    high = grasped(s, 0.160)
    for state in (target, high):
        state.update(previous_cube_height=0.145, previous_is_grasped=True)
    assert target["success"] and high["success"]
    assert compute_reward(target, ZERO) > compute_reward(high, ZERO)


def test_floor_sliding_and_excessive_motion_are_penalized():
    s = with_dist(base_state(), 0.02)
    s.update(arm_floor_contact=False, cube_xy_moved=0.0, joint_vel=ZERO.copy())
    baseline = compute_reward(s, ZERO)
    for field, value in (("arm_floor_contact", True), ("cube_xy_moved", 0.05),
                         ("joint_vel", np.full(5, 5.0))):
        unsafe = copy.deepcopy(s)
        unsafe[field] = value
        assert compute_reward(unsafe, ZERO) < baseline
    assert compute_reward(s, np.ones(5)) < baseline


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, t in tests:
        try:
            t(); print("PASS", name)
        except AssertionError:
            failed += 1; print("FAIL", name)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
