# Lab: RL pick with the SO-101 arm (MuJoCo + Stable-Baselines3)

Teach a simulated SO-101 arm to **approach, grasp and lift** a 2.5 cm cube using
reinforcement learning. No ROS 2 needed: just Python, MuJoCo, Gymnasium and SB3.

**Setup:** follow `installation.md`, then run `python scripts/check_env.py`.

## What is provided vs. what you write

| File | Who | Purpose |
|---|---|---|
| `assets/so101/` | provided | SO-101 MuJoCo model (MuJoCo Menagerie, Apache-2.0) + `pick_scene.xml` (floor + cube) |
| `so101_pick/env.py` | provided | Gymnasium env: physics, actions, contacts, resets, success check |
| `so101_pick/rewards.py` | **YOU (TODO 1)** | the reward function |
| `so101_pick/observations.py` | provided | what the policy sees (21-D, documented in the file) |
| `train.py` | **YOU (TODO 2)** + provided harness | tune the baseline hyperparameters (SAC or PPO) |
| `evaluate.py` | provided | success rate over N episodes, optional video |
| `tests/` | provided | `test_env.py` (should pass), `test_reward.py` (checks your TODO 1) |

## The task in one paragraph
Each step the policy outputs 5 numbers in [-1, 1]: change of shoulder pan, shoulder lift, elbow,
wrist flex, and gripper (20 Hz, 150 steps per episode). The cube appears on the floor in front of
the arm. **Success** = both jaws touch the cube *and* the cube centre is >= 15 cm above the floor.

## Workflow
1. `python scripts/check_env.py` and `python tests/test_env.py`: everything passes.
2. **Part A:** keep the starter reward and the baseline `HPARAMS` in `train.py`, and train briefly
   with `--randomization 0`. Look at TensorBoard: why does it not learn?
3. **Part B (TODO 1):** write the staged reward. Run `python tests/test_reward.py`, then train with
   `--randomization 0` (cube always in the same place).
4. **Part C (TODO 2):** tune the baseline hyperparameters (one change per run, different `--run-name`),
   then train with `--randomization 1` (random cube positions).
5. Evaluate: `python evaluate.py runs/<name>/best_model.zip --episodes 100 --video rollout.mp4`

```bash
python train.py --run-name my_run --timesteps 200000 --randomization 0
tensorboard --logdir runs            # Colab: %load_ext tensorboard ; %tensorboard --logdir runs
```

## Improve robustness to randomized resets

Continue from the validated fixed-layout policy in a new run:

```bash
conda activate so101
python scripts/train_randomized.py --run-name pick_randomized_v1 --timesteps 300000
```

The curriculum starts at `randomization=0.025` and advances through
`0.05, 0.1, 0.2, 0.4, 0.7, 1` after at least 80% success over 20 validation
episodes in two consecutive evaluations. It evaluates every 10,000 additional training steps and always uses
the normal home pose. Reaching the last level within the budget is not guaranteed.
Before updating, the script collects 10,000 transitions from the starting policy
to retain examples of its existing grasps in the replay buffer. These are the
policy's own actions, without expert demonstrations. The learning rate is
`3e-5`; the original validated checkpoint is preserved.

`runs/<run-name>/progress.jsonl` records results for the current level and full
randomization. `best_model.zip` and `vecnormalize_best.pkl` are selected together
by success at **full randomization**, with return breaking ties. After training,
`test_full.json` contains a separate 100-episode test using `evaluate.py`.
Each `stage_<level>/` directory also retains the best policy for that level,
with matching normalization statistics and replay buffer. This allows resuming
local progress even when full-randomization success is still low.

```bash
python evaluate.py runs/pick_randomized_v1/best_model.zip --episodes 100 --randomization 1
```

To continue an interrupted run, choose a new run name and pass its saved
`final_model.zip`, `vecnormalize_final.pkl`, and `replay_buffer.pkl` using
`--init`, `--vecnormalize`, and `--replay-buffer`. Set `--levels` to the remaining
levels (ending at `1`); use `progress.jsonl` to find the last training level.

For example, continue the completed `pick_randomized_v2` run from level `0.05`
with smaller increments and larger validation samples:

```bash
python scripts/train_randomized.py --run-name pick_randomized_v3 \
  --init runs/pick_randomized_v2/final_model.zip \
  --vecnormalize runs/pick_randomized_v2/vecnormalize_final.pkl \
  --replay-buffer runs/pick_randomized_v2/replay_buffer.pkl \
  --levels 0.05 0.075 0.1 0.15 0.2 0.3 0.4 0.6 0.8 1 \
  --timesteps 600000 --eval-freq 20000 --eval-episodes 100 --seed 43
```

If position-level training stalls, an alternative is to train across the entire
position range immediately and gradually move the initial arm pose from near
the cube back to home. Validation and final testing always start at normal home;
assisted-start training results do not count as task success.

The pilot recorded in `runs/reset_curriculum_pilot.json` found 29/40 successes
from assisted starts with the best `v3` level-0.075 checkpoint, motivating this
experiment. It uses a fresh replay buffer filled with that policy's own
full-range assisted-start trajectories:

```bash
python scripts/train_randomized.py --run-name pick_randomized_v4 \
  --init runs/pick_randomized_v3/stage_0.075/best_model.zip \
  --vecnormalize runs/pick_randomized_v3/stage_0.075/vecnormalize_best.pkl \
  --levels 1 --reset-curriculum-steps 250000 \
  --reset-curriculum-mode interpolate --prefill-steps 20000 \
  --timesteps 400000 --eval-freq 20000 --eval-episodes 100 --seed 44
```

The reset curriculum runs for 250,000 **additional** training steps regardless
of the loaded model's existing step count. The remaining 150,000 steps train
from normal home. `progress.jsonl` includes the training reset probability and
start-pose fraction (0 = near cube, 1 = home).

## Controlled learning-rate continuation

`runs/pick_v6_lr_ab/experiment.json` records a paired experiment starting from
the same frozen `pick_v5_full/best_model.zip` and matching observation statistics.
Both arms use the same raw SAC replay snapshot, training seed 0, full
randomization, and 100,000 additional steps. The control keeps `learning_rate=3e-4`;
the treatment uses `1e-4`. Network, rewards, reset behavior, and replay capacity
are unchanged. The replay was saved at the end of the source run and contains
experience from later policies; SAC supports this off-policy reuse because the
saved observations and rewards are unnormalized.

Each arm validates every 20,000 steps over 100 episodes with seed 500, selects
by success then return, and includes the final policy in that selection. The
public evaluator then tests the selected checkpoint over 200 episodes with seed
12345, matching the original evaluation command. Validation uses SB3's standard
auto-reset loop; public evaluation uses the reset loop in `evaluate.py`.

```bash
tail -f runs/pick_v6_lr_ab/control.log runs/pick_v6_lr_ab/treatment.log
```

Check `baseline_test.json`, `control/test_full.json`, and
`treatment/test_full.json` for the directly comparable public-evaluation results.
After both runs finish, `runs/pick_v6_lr_ab/finalize.py` evaluates all three
policies with a separate test seed (6001), also over 200 episodes. It writes
`comparison.json` with the validation-selected candidate, both test results,
and gained/lost episodes relative to the baseline. The comparison coordinator
is started separately and waits up to one hour for the two training results.
A small gain from this single training seed still needs further confirmation.

The completed first comparison did not improve the baseline. Its audit also
found that SB3's initial training reset had updated the cloned observation
statistics by four observations before saving the initial checkpoint. The
current runner freezes normalization through that reset and baseline evaluation,
then enables updates for training; a regression test checks exact preservation.
The original baseline files were preserved. See `comparison.json` for the
recorded distinction between newly trained validation results and retained
initial checkpoints.

## Lift-progress and overshoot reward

The reward now adds `2 * (held_height_now - held_height_before)`, where held
height is divided by the 15.1 cm goal, clipped to `[0, 1]`, and zero without
both-jaw contact. Rising while grasped earns credit; lowering or losing the
grasp removes it. Holding still earns no progress bonus. The success bonus is
now `2 + steps_left / 150`.

While grasped, excess height above 15.5 cm costs
`4 * tanh((height - 0.155) / 0.005)`. Heights use metres above the cube's resting
height. The episode still ends on the first grasped step at or above 15.1 cm,
so this penalty applies to overshoot in that terminal step. Evaluation still
counts overshoot as success under the existing task definition.

The previous grasp/height fields are supplied only to the reward; observations
remain 21-dimensional. Retraining is required to learn this behavior. Existing
replay buffers contain rewards from the old objective and should not be reused
unchanged; create a fresh buffer with the revised reward.
