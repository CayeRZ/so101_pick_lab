"""Training script (PROVIDED harness + TODO: tune the hyperparameters).

Usage examples
--------------
    python train.py --run-name first_try --timesteps 200000
    python train.py --run-name finetune --init checkpoints/warmstart.zip --timesteps 100000
    tensorboard --logdir runs        # in another terminal / Colab cell

Outputs: runs/<run-name>/  (TensorBoard logs, best_model.zip, final_model.zip, vecnormalize.pkl)

Your workflow here: run the BASELINE below once, look at the learning curves in
TensorBoard, then change ONE hyperparameter at a time and compare runs
(use a different --run-name for each experiment).
"""
import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC  # noqa: F401  (both are available to you)
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

import so101_pick  # noqa: F401  (registers "SO101Pick-v0")
from so101_pick.callbacks import (ResetCurriculumCallback, SaveNormalizationCallback,
                                 StageEvalCallback, StopOnFullSuccessCallback)
from so101_pick.env import LIFT_HEIGHT

# =============================================================================
# TODO 3a: choose the algorithm.  Set ALGO to "SAC" or "PPO".
#   SAC: off-policy, usually needs far fewer environment steps. 1 env is fine.
#   PPO: on-policy, needs many parallel envs (set N_ENVS below) and many more steps.
# The default is "SAC". In your report: justify your choice in 2-3 sentences.
# =============================================================================
ALGO = "SAC"  # <- "SAC" or "PPO"

# =============================================================================
# TODO 3b: BASELINE hyperparameters: a reasonable starting point, NOT a tuned one.
# It may learn slowly or not at all. Your job is to improve it: change one value (or few) at a
# time, re-run, and compare the learning curves (success rate, episode reward).
# Things worth testing:
#   SAC: learning_rate, batch_size, buffer_size, gamma, tau, ent_coef ("auto" or a fixed
#        value such as 0.05), learning_starts, train_freq / gradient_steps, net_arch
#   PPO: learning_rate, n_steps (per env!), batch_size, n_epochs, gamma, gae_lambda,
#        clip_range, ent_coef, net_arch
# Docs: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html  (and ppo.html)
# In your report: say what you changed, why, and what happened (with a plot).
# =============================================================================
HPARAMS_SAC = {
    "learning_rate": 3e-4,
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

HPARAMS_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,          # steps per env per update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

HPARAMS = HPARAMS_SAC if ALGO == "SAC" else HPARAMS_PPO

# TODO 3c (optional): normalise observations/rewards with VecNormalize? Set True/False and say why.
USE_VECNORMALIZE = True

N_ENVS = 4  # parallel environments. SAC: 1-4 is fine. PPO: try 8+ (more CPU cores help).


def make_env(randomization: float):
    return lambda: gym.make("SO101Pick-v0", randomization=randomization)


def build_model(venv, tb_dir: str, algorithm=ALGO, hparams=None):
    if algorithm not in ("SAC", "PPO"):
        raise ValueError("ALGO must be 'SAC' or 'PPO' at the top of train.py")
    cls = {"SAC": SAC, "PPO": PPO}[algorithm]
    return cls("MlpPolicy", venv, tensorboard_log=tb_dir, verbose=1,
               **(HPARAMS if hparams is None else hparams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--randomization", type=float, default=1.0,
                    help="0 = cube always at the same place, 1 = fully random (start low, then raise)")
    ap.add_argument("--init", default=None, help="path to a .zip checkpoint to fine-tune from")
    ap.add_argument("--replay-buffer", default=None, help="SAC replay buffer to resume alongside --init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algorithm", choices=("SAC", "PPO"), default=ALGO)
    ap.add_argument("--n-envs", type=int, default=N_ENVS)
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="CPU threads for the small policy network; avoids thread overhead")
    ap.add_argument("--subproc", action="store_true", help="use subprocesses instead of sequential vectorization")
    ap.add_argument("--eval-freq", type=int, default=10_000, help="environment steps between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--curriculum-steps", type=int, default=50_000,
                    help="fade open-gripper near-cube resets to zero over this many steps; 0 disables")
    ap.add_argument("--curriculum-mode", choices=("mixture", "interpolate"), default="mixture")
    ap.add_argument("--net-width", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction,
                    default=USE_VECNORMALIZE)
    ap.add_argument("--normalize-rewards", action="store_true",
                    help="normalize training returns; evaluation rewards remain unscaled")
    ap.add_argument("--vecnormalize", default=None, help="observation statistics to resume with --init")
    ap.add_argument("--stop-after-success-evals", type=int, default=0,
                    help="stop after this many consecutive 100%% full-task evaluations; 0 disables")
    args = ap.parse_args()
    counts = [args.n_envs, args.torch_threads, args.timesteps, args.eval_freq, args.eval_episodes]
    counts += [value for value in (args.net_width, args.batch_size) if value is not None]
    if min(counts) <= 0 or min(args.curriculum_steps, args.stop_after_success_evals) < 0:
        ap.error("step counts, environment counts and thread counts must be positive")
    torch.set_num_threads(args.torch_threads)
    normalize_observations = (args.normalize_observations or args.normalize_rewards
                              or args.vecnormalize is not None)
    if args.init and normalize_observations and not args.vecnormalize:
        ap.error("resuming normalized training requires --vecnormalize")
    hparams = dict(HPARAMS if args.algorithm == ALGO else
                   (HPARAMS_SAC if args.algorithm == "SAC" else HPARAMS_PPO))
    if args.net_width is not None:
        hparams["policy_kwargs"] = dict(net_arch=[args.net_width, args.net_width])
    if args.batch_size is not None:
        hparams["batch_size"] = args.batch_size

    out = os.path.join("runs", args.run_name)
    os.makedirs(out, exist_ok=True)
    config = {**vars(args), "hyperparameters": hparams,
              "vecnormalize": normalize_observations, "target_height": LIFT_HEIGHT}
    Path(out, "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for source in ("rewards.py", "observations.py", "env.py", "callbacks.py"):
        Path(out, source).write_text(Path("so101_pick", source).read_text())
    Path(out, "train.py").write_text(Path(__file__).read_text())

    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(make_env(args.randomization), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.curriculum_steps > 0:
        venv.env_method("set_near_cube_start_probability", 1.0)
    eval_env = make_vec_env(make_env(args.randomization), n_envs=1, seed=args.seed + 1000)
    if normalize_observations:
        venv = (VecNormalize.load(args.vecnormalize, venv) if args.vecnormalize
                else VecNormalize(venv, norm_obs=True, norm_reward=args.normalize_rewards))
        venv.training = True
        venv.norm_reward = args.normalize_rewards
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    if args.init:
        cls = {"SAC": SAC, "PPO": PPO}[args.algorithm]
        model = cls.load(args.init, env=venv, tensorboard_log=out, **{k: v for k, v in hparams.items()
                                                                       if k in ("learning_rate",)})
        print(f"Loaded checkpoint {args.init}")
        if args.replay_buffer:
            if args.algorithm != "SAC":
                ap.error("--replay-buffer requires SAC")
            model.load_replay_buffer(args.replay_buffer)
    else:
        model = build_model(venv, out, args.algorithm, hparams)

    config["initial_timesteps"] = model.num_timesteps
    config["target_total_timesteps"] = model.num_timesteps + args.timesteps if args.init else args.timesteps
    config["observation_shape"] = list(model.observation_space.shape)
    Path(out, "config.json").write_text(json.dumps(config, indent=2) + "\n")

    save_normalization = (SaveNormalizationCallback(Path(out, "vecnormalize_best.pkl"))
                          if normalize_observations else None)
    eval_cb = StageEvalCallback(eval_env, best_model_save_path=out, log_path=out,
                           callback_on_new_best=save_normalization,
                           callback_after_eval=(StopOnFullSuccessCallback(args.stop_after_success_evals)
                                                if args.stop_after_success_evals > 0 else None),
                           eval_freq=max(args.eval_freq // args.n_envs, 1),
                           n_eval_episodes=args.eval_episodes, deterministic=True)
    callbacks = [eval_cb]
    if args.curriculum_steps > 0:
        callbacks.insert(0, ResetCurriculumCallback(args.curriculum_steps, args.curriculum_mode))
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                    tb_log_name="tb", progress_bar=False, reset_num_timesteps=not bool(args.init))
    except KeyboardInterrupt:
        print("Interrupted; saving the current model and replay buffer.")
    finally:
        model.save(os.path.join(out, "final_model"))
        if args.algorithm == "SAC":
            model.save_replay_buffer(os.path.join(out, "replay_buffer.pkl"))
        if normalize_observations:
            venv.save(os.path.join(out, "vecnormalize_final.pkl"))
        venv.close()
        eval_env.close()
    print("Done. Models saved in", out)


if __name__ == "__main__":
    main()
