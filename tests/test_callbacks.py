"""Exercise stage metrics through SB3's real evaluation loop and logger."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from so101_pick.callbacks import (ResetCurriculumCallback, SaveNormalizationCallback,
                                 StageEvalCallback, StopOnFullSuccessCallback)


class StageEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def __init__(self):
        self.progress = True
        self.complete = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def set_near_cube_start_probability(self, probability):
        self.near_cube_start_probability = probability

    def set_near_cube_start_fraction(self, fraction):
        self.near_cube_start_fraction = fraction

    def step(self, action):
        self.steps += 1
        info = {
            "is_grasped": self.progress and (self.steps == 1 or (self.complete and self.steps == 3)),
            "cube_height": 0.16 if self.progress and self.steps == 3 else 0.0,
            "is_success": self.progress and self.complete and self.steps == 3,
        }
        return np.zeros(1, dtype=np.float32), 0.0, False, self.steps == 3, info


class CaptureWriter(KVWriter):
    def __init__(self):
        self.records = []

    def write(self, key_values, key_excluded, step=0):
        self.records.append(dict(key_values))


def test_stage_metrics_latch_transient_progress_and_reset_between_evaluations():
    env = DummyVecEnv([StageEnv])
    writer = CaptureWriter()
    model = SimpleNamespace(
        num_timesteps=100,
        logger=Logger(None, [writer]),
        get_env=lambda: env,
        get_vec_normalize_env=lambda: None,
        predict=lambda obs, state=None, **kwargs: (np.zeros((len(obs), 1)), state),
    )
    callback = StageEvalCallback(env, eval_freq=1, n_eval_episodes=2, verbose=0, warn=False)
    callback.init_callback(model)
    assert callback.on_step()
    first = writer.records[-1]
    assert first["eval/grasp_rate"] == first["eval/lift_rate"] == 1.0
    assert np.isclose(first["eval/max_height_mean"], 0.16)
    assert first["eval/success_rate"] == 1.0

    env.envs[0].complete = False
    assert callback.on_step()
    partial = writer.records[-1]
    assert partial["eval/grasp_rate"] == 1.0  # contact was lost before the terminal step
    assert partial["eval/lift_rate"] == partial["eval/success_rate"] == 0.0
    assert np.isclose(partial["eval/max_height_mean"], 0.16)  # height alone is insufficient

    env.envs[0].progress = False
    assert callback.on_step()
    second = writer.records[-1]
    assert second["eval/grasp_rate"] == second["eval/lift_rate"] == 0.0
    assert second["eval/max_height_mean"] == 0.0
    env.close()


def test_reset_curriculum_fades_and_does_not_modify_evaluation_env():
    train_env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(StageEnv(), max_episode_steps=10)])
    eval_env = DummyVecEnv([StageEnv])
    eval_env.set_attr("near_cube_start_probability", 0.0)
    model = SimpleNamespace(num_timesteps=0, logger=Logger(None, []), get_env=lambda: train_env)
    callback = ResetCurriculumCallback(duration=100)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    assert train_env.envs[0].unwrapped.near_cube_start_probability == 1.0
    model.num_timesteps = 50
    callback.on_step()
    assert train_env.envs[0].unwrapped.near_cube_start_probability == 0.5
    model.num_timesteps = 200
    callback.on_step()
    assert train_env.envs[0].unwrapped.near_cube_start_probability == 0.0
    assert eval_env.get_attr("near_cube_start_probability") == [0.0]
    train_env.close(); eval_env.close()


def test_relative_reset_curriculum_restarts_on_resume_and_finishes_at_home():
    for mode in ("mixture", "interpolate"):
        train = DummyVecEnv([StageEnv])
        model = SimpleNamespace(num_timesteps=833844, logger=Logger(None, []), get_env=lambda: train)
        callback = ResetCurriculumCallback(duration=100, mode=mode, relative_steps=True)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        assert train.envs[0].near_cube_start_probability == 1.0
        model.num_timesteps += 50
        callback.on_step()
        if mode == "mixture":
            assert train.envs[0].near_cube_start_probability == 0.5
        else:
            assert train.envs[0].near_cube_start_probability == 1.0
            assert train.envs[0].near_cube_start_fraction == 0.5
        model.num_timesteps += 50
        callback.on_step()
        assert train.envs[0].near_cube_start_probability == 0.0
        train.close()


def test_best_checkpoint_keeps_its_own_observation_statistics(tmp_path):
    train_env = VecNormalize(DummyVecEnv([StageEnv]), norm_reward=False)
    eval_env = VecNormalize(DummyVecEnv([StageEnv]), training=False, norm_reward=False)
    model = SAC("MlpPolicy", train_env, learning_starts=100,
                policy_kwargs=dict(net_arch=[8, 8]), verbose=0)
    model.set_logger(Logger(None, []))
    normalizer_path = tmp_path / "vecnormalize_best.pkl"
    callback = StageEvalCallback(
        eval_env, eval_freq=4, n_eval_episodes=1, verbose=0, warn=False,
        best_model_save_path=str(tmp_path),
        callback_on_new_best=SaveNormalizationCallback(normalizer_path),
    )
    model.learn(total_timesteps=8, callback=callback)
    saved = VecNormalize.load(normalizer_path, DummyVecEnv([StageEnv]))
    # Equal rewards at the second evaluation must not overwrite the first
    # checkpoint's statistics with the later training distribution.
    assert (tmp_path / "best_model.zip").exists()
    assert saved.obs_rms.count < train_env.obs_rms.count
    assert np.isclose(saved.obs_rms.count, 5.0001)
    saved.close(); train_env.close(); eval_env.close()


def test_full_success_callback_stops_learning_after_consecutive_evaluations():
    train_env = DummyVecEnv([StageEnv])
    eval_env = DummyVecEnv([StageEnv])
    model = SAC("MlpPolicy", train_env, learning_starts=100,
                policy_kwargs=dict(net_arch=[8, 8]), verbose=0)
    model.set_logger(Logger(None, []))
    callback = StageEvalCallback(eval_env, eval_freq=4, n_eval_episodes=2, verbose=0, warn=False,
                                 callback_after_eval=StopOnFullSuccessCallback(2))
    model.learn(total_timesteps=100, callback=callback)
    assert model.num_timesteps == 8
    train_env.close(); eval_env.close()
