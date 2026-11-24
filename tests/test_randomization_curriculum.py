"""Check promotion across wrappers and full-task checkpoint selection."""
from types import SimpleNamespace

import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC

from scripts.train_randomized import RandomizationCurriculum, make_env, prefill_replay


def test_randomization_update_reaches_base_env_and_changes_resets():
    env = make_env(0.0)
    env.seed(7)
    fixed = env.reset()
    env.env_method("set_randomization", 1.0)
    assert env.envs[0].unwrapped.randomization == 1.0
    env.seed(7)
    randomized = env.reset()
    assert not np.allclose(fixed, randomized)
    with pytest.raises(ValueError):
        env.env_method("set_randomization", 1.1)
    env.close()


def test_curriculum_promotes_training_only_and_selects_full_success(tmp_path, monkeypatch):
    train = VecNormalize(make_env(0.025), norm_reward=False)
    saved = []
    model = SimpleNamespace(
        num_timesteps=53844, logger=Logger(None, []),
        get_env=lambda: train, get_vec_normalize_env=lambda: train,
        save=lambda path: saved.append(path.name), save_replay_buffer=lambda path: None,
    )
    results = iter([
        (0.8, -10), (0.4, -80),  # promote at 80%; save full success=40%
        (0.5, -5), (0.2, -1),   # higher return must not replace higher success
    ])

    def measure(*args):
        success, reward = next(results)
        return dict(success_rate=success, grasp_rate=success, mean_return=reward,
                    mean_length=10, episodes=20, seed=1000)

    monkeypatch.setattr("scripts.train_randomized.measure", measure)
    callback = RandomizationCurriculum(tmp_path, [0.025, 0.05, 1.0], 10000, 20, 0.8,
                                      promotion_evals=1)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    assert train.envs[0].unwrapped.randomization == 0.05
    assert callback.stage_env.envs[0].unwrapped.randomization == 0.05
    assert callback.full_env.envs[0].unwrapped.randomization == 1.0
    # The resumed global timestep must not trigger an evaluation on every step.
    assert callback.last_eval == 53844
    model.num_timesteps += 10000
    callback.on_step()
    assert saved.count("best_model.zip") == 3  # full-task best and one per stage
    assert callback.stage == 1
    assert (tmp_path / "vecnormalize_best.pkl").exists()
    callback.close()
    train.close()


def test_promotion_requires_consecutive_success_and_preserves_stage_best(tmp_path, monkeypatch):
    train = VecNormalize(make_env(0.05), norm_reward=False)
    model = SimpleNamespace(
        num_timesteps=353844, logger=Logger(None, []),
        get_env=lambda: train, get_vec_normalize_env=lambda: train,
        save=lambda path: None, save_replay_buffer=lambda path: path.touch(),
    )
    stage_successes = iter([0.9, 0.9, 0.6, 0.85, 0.8])

    def measure(model, env, episodes, seed):
        success = next(stage_successes) if env.envs[0].unwrapped.randomization == 0.05 else 0.0
        return dict(success_rate=success, grasp_rate=success, mean_return=-100.0)

    monkeypatch.setattr("scripts.train_randomized.measure", measure)
    callback = RandomizationCurriculum(tmp_path, [0.05, 0.075, 1.0], 10000, 20, 0.8)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    assert callback.stage == 0  # one good evaluation cannot promote
    callback._evaluate()
    assert callback.stage == 0  # repeating the same timestep cannot promote
    assert callback.success_streak == 1
    for _ in range(2):
        model.num_timesteps += 10000
        callback.on_step()
        assert callback.stage == 0  # failed evaluation resets the streak
    model.num_timesteps += 10000
    callback.on_step()
    assert callback.stage == 1
    assert train.envs[0].unwrapped.randomization == 0.075
    import json
    record = json.loads((tmp_path / "stage_0.05" / "validation.json").read_text())
    assert record["current"]["success_rate"] == 0.9
    assert (tmp_path / "stage_0.05" / "replay_buffer.pkl").exists()
    callback.close()
    train.close()


def test_prefill_stores_raw_terminal_observation_before_auto_reset(monkeypatch):
    class TerminalEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.ones(1, dtype=np.float32), 2.0, True, False, {"is_success": True}

    monkeypatch.setattr("scripts.train_randomized.make_env",
                        lambda *args, **kwargs: DummyVecEnv([TerminalEnv] * 4))
    transitions = []

    def predict(obs, **kwargs):
        assert np.all(obs == 10)  # actor gets normalized input
        return np.zeros((4, 1)), None

    model = SimpleNamespace(predict=predict,
                            policy=SimpleNamespace(scale_action=lambda action: action),
                            replay_buffer=SimpleNamespace(add=lambda *args: transitions.append(args)))
    result = prefill_replay(model, SimpleNamespace(normalize_obs=lambda obs: obs + 10),
                           0.025, steps=8, seed=42)
    assert result == dict(transitions=8, episodes=8, successes=8)
    assert len(transitions) == 2
    for obs, next_obs, action, rewards, dones, infos in transitions:
        assert np.all(obs == 0)  # replay uses original, unnormalized input
        assert np.all(next_obs == 1)  # terminal state, not auto-reset state
        assert np.all(rewards == 2)


def test_assisted_training_starts_never_leak_into_validation(tmp_path, monkeypatch):
    train = VecNormalize(make_env(1.0), norm_reward=False)
    train.env_method("set_near_cube_start_probability", 1.0)
    train.env_method("set_near_cube_start_fraction", 0.5)
    model = SimpleNamespace(
        num_timesteps=833844, logger=Logger(None, []),
        get_env=lambda: train, get_vec_normalize_env=lambda: train,
        save=lambda path: None, save_replay_buffer=lambda path: None,
    )

    def measure(model, env, episodes, seed):
        assert env.envs[0].unwrapped.near_cube_start_probability == 0.0
        assert seed == 321
        return dict(success_rate=0.0, grasp_rate=0.0, mean_return=-100.0)

    monkeypatch.setattr("scripts.train_randomized.measure", measure)
    callback = RandomizationCurriculum(tmp_path, [1.0], 10000, 20, 0.8, validation_seed=321)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    import json
    record = json.loads((tmp_path / "best_validation.json").read_text())
    assert record["training_near_cube_probability"] == 1.0
    assert record["training_start_fraction"] == 0.5
    callback.close()
    train.close()


def test_initial_checkpoint_preserves_normalization_through_sb3_training_reset(tmp_path, monkeypatch):
    train = VecNormalize(make_env(1.0, n_envs=4), norm_reward=False)
    train.seed(42)
    train.reset()
    source_count = train.obs_rms.count
    source_mean = train.obs_rms.mean.copy()
    model = SAC("MlpPolicy", train, learning_starts=100, verbose=0,
                policy_kwargs=dict(net_arch=[8, 8]))
    model.set_logger(Logger(None, []))
    monkeypatch.setattr("scripts.train_randomized.measure", lambda *args: dict(
        success_rate=0.5, grasp_rate=0.5, mean_return=-100.0))
    callback = RandomizationCurriculum(tmp_path, [1.0], 10000, 1, 0.8)
    train.training = False
    model.learn(total_timesteps=8, callback=callback)
    saved = VecNormalize.load(tmp_path / "vecnormalize_best.pkl", make_env(1.0))
    assert saved.obs_rms.count == source_count
    np.testing.assert_array_equal(saved.obs_rms.mean, source_mean)
    assert train.training
    assert train.obs_rms.count == source_count + 8
    saved.close()
    callback.close()
    train.close()
