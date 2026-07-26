"""Device, reproducibility, checkpoint, and learning-rate scheduling utilities."""

import os
from collections.abc import Callable, MutableMapping
from datetime import datetime
from typing import cast

import numpy as np
import pytz
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


def torch_get_device(device_type: str) -> torch.device:
    """Resolve the requested torch device.

    Args:
        device_type: Requested device type: ``cuda``, ``auto``, or CPU.

    Returns:
        torch.device: Selected execution device.
    """
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available :(, `python train.py +device=auto`"
        device = torch.device("cuda")
    elif device_type == "auto":
        assert not torch.cuda.is_available(), "CUDA is available :), switch to cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                device = xm.xla_device()
            except ImportError:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def torch_set_seed(seed: int) -> None:
    """Seed torch RNGs and configure deterministic cuDNN behavior.

    Args:
        seed: Seed applied to CPU and available CUDA random number generators.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_compile_ckpt_fix(state_dict: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    """Strip ``torch.compile`` parameter prefixes from a checkpoint state dict.

    Args:
        state_dict: Checkpoint parameter mapping to update in place.

    Returns:
        MutableMapping[str, torch.Tensor]: The updated state dict with
            compile-specific prefixes removed.
    """
    # when torch.compiled a model, state_dict is updated with a prefix '_orig_mod.', renaming this
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def get_ist_time_now(fmt: str = "%d-%m-%Y-%H%M%S") -> str:
    """Format the current India Standard Time timestamp.

    Args:
        fmt: ``datetime.strftime`` format string.

    Returns:
        str: Formatted IST timestamp.
    """
    tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(tz)
    return now_ist.strftime(fmt)


def _cosine_scheduler(start_val: float, end_val: float, steps: int) -> np.ndarray:
    """Create a cosine decay schedule between two values.

    Args:
        start_val: Initial schedule value.
        end_val: Final schedule value.
        steps: Number of scheduled steps.

    Returns:
        np.ndarray: Schedule values for each step.
    """
    iters = np.arange(steps)
    schedule = end_val + 0.5 * (start_val - end_val) * (1 + np.cos(np.pi * iters / steps))
    return schedule


def cosine_scheduler(start_val: float, end_val: float, steps: int, total_steps: int) -> Callable[[int], float]:
    """Create a callable cosine decay schedule.

    Args:
        start_val: Initial schedule value.
        end_val: Final schedule value.
        steps: Number of cosine-decay steps.
        total_steps: Total number of training steps.

    Returns:
        Callable[[int], float]: Function that returns the value at a step.
    """
    schedule = np.concatenate([_cosine_scheduler(start_val, end_val, steps), np.full(total_steps - steps, end_val)])
    return lambda current_step: float(schedule[current_step])


def linear_cosine_constant_scheduler(
    start_val: float, max_val: float, min_val: float, total_steps: int, linear_steps: int, cosine_steps: int
) -> Callable[[int], float]:
    """Create a linear warmup, cosine decay, then constant schedule.

    Args:
        start_val: Initial value before warmup.
        max_val: Peak value after warmup.
        min_val: Value reached after cosine decay.
        total_steps: Total number of training steps.
        linear_steps: Number of warmup steps.
        cosine_steps: Number of cosine-decay steps.

    Returns:
        Callable[[int], float]: Function that returns the value at a step.
    """
    assert linear_steps + cosine_steps <= total_steps, (
        f"linear_steps + cosine_steps must be <= total_steps, got {linear_steps + cosine_steps} vs {total_steps}"
    )
    linear_schedule = np.linspace(start_val, max_val, linear_steps + 1)[
        1:
    ]  # so that first value > min_val (in case min_val == 0)
    cosine_schedule = _cosine_scheduler(max_val, min_val, cosine_steps)
    constant_schedule = np.full(total_steps - linear_steps - cosine_steps, min_val)
    schedule = np.concatenate([linear_schedule, cosine_schedule, constant_schedule])
    return lambda current_step: float(schedule[current_step])


def constant_scheduler(val: float, total_steps: int) -> Callable[[int], float]:
    """Create a constant schedule.

    Args:
        val: Value returned at every step.
        total_steps: Total number of training steps; retained for API consistency.

    Returns:
        Callable[[int], float]: Function that returns ``val`` at any step.
    """
    return lambda current_step: val


def linear_scheduler(start_val: float, end_val: float, steps: int, total_steps: int) -> Callable[[int], float]:
    """Create a linear schedule followed by a constant final value.

    Args:
        start_val: Initial schedule value.
        end_val: Final schedule value.
        steps: Number of linear-interpolation steps.
        total_steps: Total number of training steps.

    Returns:
        Callable[[int], float]: Function that returns the value at a step.
    """
    schedule = np.concatenate([np.linspace(start_val, end_val, steps + 1)[1:], np.full(total_steps - steps, end_val)])
    return lambda current_step: float(schedule[current_step])


def init_scheduler(cfg: DictConfig, batch_per_epoch: int, n_epochs: int) -> Callable[[int], float]:
    """Build a step or epoch indexed schedule from the configuration.

    Args:
        cfg: Scheduler configuration, including ``type`` and ``mode``.
        batch_per_epoch: Number of batches in each epoch.
        n_epochs: Total number of epochs.

    Returns:
        Callable[[int], float]: Function that returns the value at a schedule index.
    """
    scheduler_map = {
        "linear_cosine_constant": linear_cosine_constant_scheduler,
        "cosine": cosine_scheduler,
        "linear": linear_scheduler,
        "constant": constant_scheduler,
    }
    kwargs = cast(dict[str, object], dict(cfg))
    mode = kwargs.pop("mode")
    assert mode in ("steps", "epochs"), f"Invalid mode: {mode}"
    scheduler_type = str(kwargs.pop("type"))
    assert scheduler_type in scheduler_map, f"Unknown scheduler type: {scheduler_type}"
    scheduler_fn = scheduler_map[scheduler_type]
    batch_per_epoch = 1 if mode == "epochs" else batch_per_epoch
    for key in list(kwargs):
        if key.endswith("epochs"):
            kwargs[key.replace("epochs", "steps")] = cast(int | float, kwargs.pop(key)) * batch_per_epoch
    kwargs["total_steps"] = batch_per_epoch * n_epochs
    return scheduler_fn(**kwargs)


class WandBLogger:
    """Optionally log scalar metrics to Weights & Biases."""

    def __init__(
        self,
        project: str,
        run_id: str,
        config: DictConfig,
        enable: bool = False,
        step_metric: str = "epoch",
    ) -> None:
        """Initialize a W&B run when logging is enabled.

        Args:
            project: W&B project name.
            run_id: Name assigned to the run.
            config: Hydra configuration saved with the run.
            enable: Whether to initialize and log to W&B.
            step_metric: Metric used as the x-axis for logged values.
        """
        self.run_id = run_id
        self.enable = enable
        self.defined_metrics: set[str] = set()
        self.step_metric = step_metric
        if self.enable:
            load_dotenv()
            wandb_key = os.getenv("WANDB_API_KEY")
            assert wandb_key is not None, "WANDB_API_KEY not loaded in env"
            wandb.login(key=wandb_key)
            wandb_config = cast(dict[str, object], OmegaConf.to_container(config, resolve=True))
            wandb.init(project=project, name=run_id, config=wandb_config)
            wandb.define_metric(self.step_metric)
            self.defined_metrics.add(self.step_metric)

    def __call__(self, metrics: dict[str, float]) -> None:
        """Log metrics to W&B when enabled.

        Args:
            metrics: Scalar metrics keyed by metric name.
        """
        if not self.enable:
            return
        for metric in metrics:
            if metric not in self.defined_metrics:
                wandb.define_metric(metric, step_metric=self.step_metric)
                self.defined_metrics.add(metric)
        wandb.log(metrics)


def format_metrics(metrics: dict[str, float], epoch: int, n_epochs: int) -> str:
    """Format epoch metrics for text logging.

    Args:
        metrics: Scalar metrics keyed by metric name.
        epoch: Current one-based epoch number.
        n_epochs: Total number of epochs.

    Returns:
        str: Single-line metric summary.
    """
    metric_text = " ".join(
        f"{key}={value:.2e}"
        if "lr" in key.lower()
        else f"{key}={value:.2%}"
        if "acc" in key.lower()
        else f"{key}={value / 60:.2f}m"
        if "time" in key.lower()
        else f"{key}={value:.4f}"
        for key, value in metrics.items()
        if key != "epoch"
    )
    return f"Epoch {epoch}/{n_epochs} {metric_text}"
