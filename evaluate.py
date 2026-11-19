"""Evaluate a trained policy: success rate over N episodes, optionally record a video.
(PROVIDED; this is also how your result is graded.)

    python evaluate.py runs/first_try/best_model.zip --episodes 100
    python evaluate.py runs/first_try/best_model.zip --episodes 5 --video rollout.mp4
    (add --vecnormalize runs/first_try/vecnormalize.pkl if you trained with VecNormalize)
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import so101_pick  # noqa: F401
from so101_pick.env import LIFT_HEIGHT


class TerminalFrameCapture(gym.Wrapper):
    """Capture the last frame before DummyVecEnv automatically resets."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["terminal_frame"] = self.env.render()
        return obs, reward, terminated, truncated, info


def load(path):
    for cls in (SAC, PPO):
        try:
            return cls.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"Could not load {path} as SAC or PPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--randomization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None, help="write the first few episodes to this .mp4")
    ap.add_argument("--vecnormalize", default=None)
    ap.add_argument("--json", dest="json_path", help="save per-episode measurements as JSON")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()
    if args.episodes <= 0 or args.torch_threads <= 0:
        ap.error("--episodes and --torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    render = args.video is not None
    if args.vecnormalize is None:
        model_path = Path(args.model)
        suffix = {"best_model": "best", "final_model": "final"}.get(model_path.stem)
        candidate = model_path.parent / f"vecnormalize_{suffix}.pkl"
        if suffix and candidate.is_file():
            args.vecnormalize = str(candidate)
    def make_eval_env():
        env = gym.make("SO101Pick-v0", randomization=args.randomization,
                       render_mode="rgb_array" if render else None)
        return TerminalFrameCapture(env) if render else env

    venv = DummyVecEnv([make_eval_env])
    if args.vecnormalize:
        venv = VecNormalize.load(args.vecnormalize, venv)
        venv.training = False
        venv.norm_reward = False
    model = load(args.model)
    venv.seed(args.seed)

    successes, grasped, lifted, frames = 0, 0, 0, []
    measurements = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done, ever_grasped, ever_lifted = False, False, False
        peak_height, peak_grasp_height, min_distance = 0.0, 0.0, float("inf")
        steps, floor_steps, opening_sum, episode_return = 0, 0, 0.0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done_arr, infos = venv.step(action)
            done = bool(done_arr[0])
            info = infos[0]
            ever_grasped |= bool(info.get("is_grasped", False))
            ever_lifted |= bool(info.get("is_success", False))
            steps += 1
            episode_return += float(rewards[0])
            floor_steps += bool(info.get("arm_floor_contact", False))
            opening_sum += float(info.get("gripper_opening", 0.0))
            peak_height = max(peak_height, float(info["cube_height"]))
            if info["is_grasped"]:
                peak_grasp_height = max(peak_grasp_height, float(info["cube_height"]))
            min_distance = min(min_distance, float(info["dist_ee_cube"]))
            if render and ep < 5:
                frames.append(info["terminal_frame"] if done else venv.envs[0].unwrapped.render())
        successes += int(info.get("is_success", False))
        grasped += int(ever_grasped)
        lifted += int(ever_lifted)
        measurements.append(dict(success=bool(info["is_success"]), grasped=ever_grasped,
                                 peak_height=peak_height, peak_grasp_height=peak_grasp_height,
                                 min_distance=min_distance, steps=steps, floor_steps=floor_steps,
                                 mean_opening=opening_sum / steps, episode_return=episode_return))

    print(f"episodes: {args.episodes}")
    height_cm = f"{LIFT_HEIGHT * 100:g}"
    print(f"success rate (grasped + lifted {height_cm} cm): {successes / args.episodes:.1%}")
    print(f"grasped at least once:                {grasped / args.episodes:.1%}")
    print(f"lifted {height_cm} cm at least once:           {lifted / args.episodes:.1%}")
    print(f"mean peak height while grasped:       {100 * np.mean([m['peak_grasp_height'] for m in measurements]):.2f} cm")
    if args.json_path:
        report = dict(model=args.model, model_timesteps=int(model.num_timesteps),
                      target_height=LIFT_HEIGHT, randomization=args.randomization, seed=args.seed,
                      success_rate=successes / args.episodes, grasp_rate=grasped / args.episodes,
                      episodes=measurements)
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
    if render and frames:
        imageio.mimsave(args.video, frames, fps=20)
        print("video written to", args.video)
    venv.close()


if __name__ == "__main__":
    main()
