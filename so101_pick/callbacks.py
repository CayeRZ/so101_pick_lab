"""Evaluation metrics for each stage of the grasp-and-lift task."""

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback


class SaveNormalizationCallback(BaseCallback):
    """Save observation statistics at the same time as the best model."""

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def _on_step(self):
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is not None:
            normalizer.save(self.path)
        return True


class StopOnFullSuccessCallback(BaseCallback):
    """Stop only after consecutive all-success evaluations from normal home."""

    def __init__(self, evaluations: int):
        super().__init__()
        self.evaluations = evaluations
        self.streak = 0

    def _on_step(self):
        successes = self.parent._is_success_buffer
        self.streak = self.streak + 1 if successes and all(successes) else 0
        if self.streak >= self.evaluations:
            print(f"Stopping after {self.streak} consecutive 100% success evaluations.")
            return False
        return True


class ResetCurriculumCallback(BaseCallback):
    """Fade near-cube resets to zero; evaluation environments are untouched."""

    def __init__(self, duration: int, mode: str = "mixture", relative_steps: bool = False):
        super().__init__()
        if duration <= 0:
            raise ValueError("curriculum duration must be positive")
        self.duration = duration
        if mode not in ("mixture", "interpolate"):
            raise ValueError("unknown curriculum mode")
        self.mode = mode
        self.relative_steps = relative_steps
        self._start_timestep = 0
        self._last_update_step = None
        self._last_setting = None

    def _on_training_start(self):
        self._start_timestep = self.num_timesteps if self.relative_steps else 0
        self._update_probability()

    def _update_probability(self):
        fraction = np.clip((self.num_timesteps - self._start_timestep) / self.duration, 0.0, 1.0)
        probability = (1.0 - fraction if self.mode == "mixture" else float(fraction < 1.0))
        setting = (probability, fraction)
        interval = min(100, max(1, self.duration // 100))
        due = self._last_update_step is None or self.num_timesteps - self._last_update_step >= interval
        if setting != self._last_setting and (due or probability == 0.0):
            self.training_env.env_method("set_near_cube_start_probability", probability)
            if self.mode == "interpolate":
                self.training_env.env_method("set_near_cube_start_fraction", fraction)
            self._last_update_step = self.num_timesteps
            self._last_setting = setting
        self.logger.record("curriculum/near_cube_probability", probability)
        if self.mode == "interpolate":
            self.logger.record("curriculum/start_distance_fraction", fraction)

    def _on_step(self):
        self._update_probability()
        return True


class StageEvalCallback(EvalCallback):
    """Keep SB3 evaluation and report grasp/lift rates before full success."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_stage_metrics()

    def _reset_stage_metrics(self):
        self._episodes = {}
        self._completed = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._reset_stage_metrics()
        return super()._on_step()

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        # Evaluation can interleave multiple vectorized environments.
        index = locals_.get("i", 0)
        episode = self._episodes.setdefault(index, [False, False, 0.0])
        episode[0] |= bool(info.get("is_grasped", False))
        episode[1] |= bool(info.get("is_success", False))
        episode[2] = max(episode[2], float(info.get("cube_height", 0.0)))
        if locals_["done"]:
            self._completed.append(self._episodes.pop(index))
            grasp, lift, height = np.mean(self._completed, axis=0)
            # Record before EvalCallback dumps this evaluation's logger.
            self.logger.record("eval/grasp_rate", float(grasp))
            self.logger.record("eval/lift_rate", float(lift))
            self.logger.record("eval/max_height_mean", float(height))
