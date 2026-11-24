"""Fine-tune with a position curriculum or an assisted-reset curriculum.

Run from the repository root using the so101 Conda environment:
    python scripts/train_randomized.py --run-name pick_randomized_v1

All evaluations start at normal home, without the near-cube reset curriculum.
The best checkpoint is selected by full-randomization success, then return.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

import so101_pick  # noqa: F401
from so101_pick.env import LIFT_HEIGHT
from so101_pick.callbacks import ResetCurriculumCallback


def make_env(randomization, n_envs=1):
    return DummyVecEnv([
        lambda: Monitor(gym.make("SO101Pick-v0", randomization=randomization))
        for _ in range(n_envs)
    ])


def measure(model, env, episodes, seed):
    """Use a repeatable validation set; final testing uses a different seed."""
    successes, grasps = [], []
    ever_grasped = False

    def capture(locals_, globals_):
        nonlocal ever_grasped
        ever_grasped |= bool(locals_["info"].get("is_grasped", False))
        if locals_["done"]:
            successes.append(bool(locals_["info"]["is_success"]))
            grasps.append(ever_grasped)
            ever_grasped = False

    env.seed(seed)
    returns, lengths = evaluate_policy(
        model, env, n_eval_episodes=episodes, deterministic=True,
        return_episode_rewards=True, callback=capture,
    )
    return dict(success_rate=float(np.mean(successes)), grasp_rate=float(np.mean(grasps)),
                mean_return=float(np.mean(returns)), mean_length=float(np.mean(lengths)),
                episodes=episodes, seed=seed)


def prefill_replay(model, normalizer, randomization, steps, seed, near_cube_probability=0.0):
    """Retain the policy's own experience before fine-tuning a fresh buffer.

    SB3 replay stores unnormalized observations/rewards and terminal observations
    before vectorized auto-reset. No expert or hand-written actions are used.
    """
    env = make_env(randomization, n_envs=4)
    if near_cube_probability:
        env.env_method("set_near_cube_start_probability", near_cube_probability)
    env.seed(seed)
    obs = env.reset()
    successes, episodes = 0, 0
    try:
        for _ in range((steps + env.num_envs - 1) // env.num_envs):
            action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
            next_obs, rewards, dones, infos = env.step(action)
            terminal_obs = next_obs.copy()
            for index, done in enumerate(dones):
                if done:
                    terminal_obs[index] = infos[index]["terminal_observation"]
                    episodes += 1
                    successes += bool(infos[index]["is_success"])
            model.replay_buffer.add(obs, terminal_obs, model.policy.scale_action(action),
                                    rewards, dones, infos)
            obs = next_obs
    finally:
        env.close()
    result = dict(transitions=((steps + 3) // 4) * 4, episodes=episodes, successes=successes)
    print("Prefilled replay from the initial policy:", result, flush=True)
    return result


class RandomizationCurriculum(BaseCallback):
    def __init__(self, out, levels, eval_freq, episodes, threshold, promotion_evals=2,
                 validation_seed=1000):
        super().__init__()
        self.out, self.levels = out, levels
        self.eval_freq, self.episodes, self.threshold = eval_freq, episodes, threshold
        self.validation_seed = validation_seed
        self.stage = 0
        self.promotion_evals = promotion_evals
        self.success_streak = 0
        self.last_promotion_step = None
        self.stage_best_scores = {}
        self.best_score = (-1.0, -float("inf"))
        self.last_eval = 0
        self.stage_env = VecNormalize(make_env(levels[0]), training=False, norm_reward=False)
        self.full_env = VecNormalize(make_env(1.0), training=False, norm_reward=False)

    def _evaluate(self):
        normalizer = self.model.get_vec_normalize_env()
        for env in (self.stage_env, self.full_env):
            sync_envs_normalization(self.training_env, env)
        current = measure(self.model, self.stage_env, self.episodes, self.validation_seed)
        full = (current if self.levels[self.stage] == 1.0 else
                measure(self.model, self.full_env, self.episodes, self.validation_seed))
        record = dict(timesteps=self.num_timesteps, randomization=self.levels[self.stage],
                      current=current, full=full,
                      training_near_cube_probability=self.training_env.get_attr("near_cube_start_probability")[0],
                      training_start_fraction=self.training_env.get_attr("near_cube_start_fraction")[0])
        with (self.out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        for name, result in (("curriculum", current), ("full_randomization", full)):
            for key in ("success_rate", "grasp_rate", "mean_return"):
                self.logger.record(f"{name}/{key}", result[key])
        self.logger.record("curriculum/randomization", self.levels[self.stage])
        self.logger.dump(self.num_timesteps)
        print(f"Step {self.num_timesteps}: randomization={self.levels[self.stage]:g}, "
              f"stage success={current['success_rate']:.0%}, "
              f"full success={full['success_rate']:.0%}", flush=True)
        score = (full["success_rate"], full["mean_return"])
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.out / "best_model.zip")
            normalizer.save(self.out / "vecnormalize_best.pkl")
            (self.out / "best_validation.json").write_text(json.dumps(record, indent=2) + "\n")
        # A policy good at the current level is useful for resuming even when
        # it has not yet generalized to full randomization.
        stage_score = (current["success_rate"], current["mean_return"])
        if stage_score > self.stage_best_scores.get(self.stage, (-1.0, -float("inf"))):
            self.stage_best_scores[self.stage] = stage_score
            stage_out = self.out / f"stage_{self.levels[self.stage]:g}"
            stage_out.mkdir(exist_ok=True)
            self.model.save(stage_out / "best_model.zip")
            normalizer.save(stage_out / "vecnormalize_best.pkl")
            self.model.save_replay_buffer(stage_out / "replay_buffer.pkl")
            (stage_out / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
        # Always retain a recent matched model/statistics/buffer set for resumption.
        self.model.save(self.out / "latest_model.zip")
        normalizer.save(self.out / "vecnormalize_latest.pkl")
        self.model.save_replay_buffer(self.out / "replay_buffer.pkl")
        # A final evaluation can repeat the last training timestep. It must not
        # count as a second independent success for curriculum promotion.
        if self.num_timesteps != self.last_promotion_step:
            self.success_streak = (self.success_streak + 1
                                   if current["success_rate"] >= self.threshold else 0)
            self.last_promotion_step = self.num_timesteps
        if self.success_streak >= self.promotion_evals and self.stage < len(self.levels) - 1:
            self.stage += 1
            self.success_streak = 0
            level = self.levels[self.stage]
            for env in (self.training_env, self.stage_env):
                env.env_method("set_randomization", level)
            print(f"Advancing to randomization={level:g}", flush=True)

    def _on_training_start(self):
        self.last_eval = self.num_timesteps
        self._evaluate()
        # main() freezes statistics through SB3's initial reset, so this first
        # checkpoint keeps the source policy's exact input normalization.
        self.model.get_vec_normalize_env().training = True

    def _on_step(self):
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._evaluate()
        return True

    def close(self):
        self.stage_env.close()
        self.full_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="pick_randomized_v1")
    parser.add_argument("--init", type=Path,
                        default=Path("runs/pick_lift_15_1cm_validated/best_model.zip"))
    parser.add_argument("--vecnormalize", type=Path,
                        default=Path("runs/pick_lift_15_1cm_validated/vecnormalize_best.pkl"))
    parser.add_argument("--replay-buffer", type=Path, default=None)
    parser.add_argument("--prefill-steps", type=int, default=10_000)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--promotion-evals", type=int, default=2,
                        help="consecutive successful evaluations required before advancing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=12345)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--reset-curriculum-steps", type=int, default=0,
                        help="fade assisted resets over this many additional steps; 0 disables")
    parser.add_argument("--reset-curriculum-mode", choices=("mixture", "interpolate"),
                        default="interpolate")
    args = parser.parse_args()
    if min(args.timesteps, args.eval_freq, args.eval_episodes, args.promotion_evals,
           args.test_episodes) <= 0 or args.prefill_steps < 0:
        parser.error("step counts and episode counts must be positive")
    if not 0 <= args.reset_curriculum_steps < args.timesteps:
        parser.error("reset curriculum must end before the training budget")
    if (not 0 < args.learning_rate or not 0 < args.threshold <= 1
            or args.levels != sorted(set(args.levels))
            or not 0 <= args.levels[0] <= args.levels[-1] == 1):
        parser.error("levels must increase within [0, 1] and end at 1; check rate/threshold")
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    venv = VecNormalize.load(args.vecnormalize, make_env(args.levels[0], n_envs=4))
    venv.training, venv.norm_reward = True, False
    venv.seed(args.seed)
    model = SAC.load(args.init, env=venv, device="cpu", seed=args.seed,
                     learning_rate=args.learning_rate, tensorboard_log=str(out))
    if args.replay_buffer:
        model.load_replay_buffer(args.replay_buffer)
        model.learning_starts = 0
        prefill = None
    elif args.prefill_steps:
        prefill = prefill_replay(model, venv, args.levels[0], args.prefill_steps, args.seed + 1,
                                 near_cube_probability=float(args.reset_curriculum_steps > 0))
        model.learning_starts = 0
    else:
        # Collect 1,000 transitions before updating on a fresh replay buffer;
        # global timesteps continue from the loaded model.
        model.learning_starts = model.num_timesteps + 1000
        prefill = None
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(initial_timesteps=model.num_timesteps, target_height=LIFT_HEIGHT,
                  target_total_timesteps=model.num_timesteps + args.timesteps,
                  n_envs=4, prefill=prefill)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for source in (Path(__file__), Path("so101_pick/env.py"),
                   Path("so101_pick/rewards.py"), Path("so101_pick/observations.py"),
                   Path("so101_pick/callbacks.py")):
        (out / source.name).write_text(source.read_text())
    callback = RandomizationCurriculum(out, args.levels, args.eval_freq,
                                      args.eval_episodes, args.threshold, args.promotion_evals,
                                      args.validation_seed)
    callbacks = [callback]
    if args.reset_curriculum_steps:
        callbacks.insert(0, ResetCurriculumCallback(args.reset_curriculum_steps,
                                                   args.reset_curriculum_mode, relative_steps=True))
    # SB3 resets the training environment before on_training_start. Do not let
    # that reset alter the source normalizer before its baseline evaluation.
    venv.training = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), reset_num_timesteps=False,
                    log_interval=100)
        callback._evaluate()
    except KeyboardInterrupt:
        print("Interrupted; saving the current training state.", flush=True)
    finally:
        model.save(out / "final_model.zip")
        venv.save(out / "vecnormalize_final.pkl")
        model.save_replay_buffer(out / "replay_buffer.pkl")
        callback.close()
        venv.close()
    # Reuse the public evaluator, including its episode reset/seed semantics.
    subprocess.run([sys.executable, "evaluate.py", str(out / "best_model.zip"),
                    "--episodes", str(args.test_episodes), "--randomization", "1",
                    "--seed", str(args.test_seed),
                    "--json", str(out / "test_full.json")], check=True)


if __name__ == "__main__":
    main()
