"""Train a DINO model with Hydra-configured data, schedules, and checkpoints."""

import logging
import math
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from time import time
from typing import cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data import init_dataloader, init_dataset
from dino import Loss, Model, ModelConfig, ema_update_teacher
from utils import (
    WandBLogger,
    format_metrics,
    get_ist_time_now,
    init_scheduler,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
)

OmegaConf.register_new_resolver("now_ist", get_ist_time_now)
logger = logging.getLogger("train")


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run DINO training from a Hydra configuration.

    Args:
        cfg: Resolved Hydra configuration for data, model, and optimization.
    """
    OmegaConf.resolve(cfg)
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    wandb_logger = WandBLogger(
        project=str(cfg.logging.wandb.project),
        run_id=str(hydra_cfg.job.name),
        config=cfg,
        enable=bool(cfg.logging.wandb.enable),
    )
    torch_autocast_dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[cfg.autocast_dtype]
    torch_set_seed(cfg.rng_seed)

    dataset = init_dataset(cfg, split="train", train_mode=True)
    dataloader = init_dataloader(dataset, cfg, train_mode=True)
    n_epochs = cfg.n_epochs
    batch_per_epoch = len(dataloader)
    logger.info(f"{n_epochs=} {batch_per_epoch=} steps={n_epochs * batch_per_epoch}")
    assert batch_per_epoch > 0, "Dataloader must contain at least one batch"

    start_epoch = 1
    init_from = str(getattr(cfg, "init_from", "scratch"))
    model_cfg = ModelConfig(**cfg.model)
    ckpt = None
    if init_from != "scratch":
        logger.info(f"Loading checkpoint from {init_from}")
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        model_cfg = ModelConfig(**ckpt["cfg"]["model"])
        start_epoch = ckpt["epoch"] + 1

    # model init
    student_model = Model(model_cfg)
    # no regularization for teacher model
    teacher_model = Model(replace(model_cfg, drop_path_rate=0.0, attn_drop_rate=0.0, drop_rate=0.0))
    if ckpt is not None:
        teacher_model.load_state_dict(torch_compile_ckpt_fix(ckpt["teacher_model"]))
        student_model.load_state_dict(torch_compile_ckpt_fix(ckpt["student_model"]))
    else:
        # start with the same weights
        teacher_model.load_state_dict(student_model.state_dict())
    # remove gradients from teacher model as it is trained as momentum encoder
    teacher_model.requires_grad_(False)
    logger.info(f"Model: {model_cfg.name} " + student_model.params_breakdown())

    decay_params = [p for p in student_model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in student_model.parameters() if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params, "apply_decay": True},
        {"params": no_decay_params, "apply_decay": False},
    ]
    optimizer_map = {
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    optimizer = optimizer_map[cfg.optimizer.type](param_groups)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for new_group, group in zip(optimizer.param_groups, param_groups, strict=True):
            new_group["apply_decay"] = group["apply_decay"]

    lr_scheduler = init_scheduler(cfg.optimizer.lr_schedule, batch_per_epoch, n_epochs)
    weight_decay_scheduler = init_scheduler(cfg.optimizer.weight_decay_schedule, batch_per_epoch, n_epochs)
    momentum_scheduler = init_scheduler(cfg.momentum_schedule, batch_per_epoch, n_epochs)
    student_temp_scheduler = init_scheduler(cfg.student_temp_schedule, batch_per_epoch, n_epochs)
    teacher_temp_scheduler = init_scheduler(cfg.teacher_temp_schedule, batch_per_epoch, n_epochs)

    loss_fn = Loss(
        out_dim=model_cfg.out_dim,
        center_momentum=cfg.center_momentum,
        n_local_crops=cfg.n_local_crops,
        n_global_crops=cfg.n_global_crops,
    )
    if ckpt is not None:
        loss_fn.load_state_dict(torch_compile_ckpt_fix(ckpt["dino_loss"]))

    student_model.to(device)
    teacher_model.to(device)
    loss_fn.to(device)

    if cfg.torch_compile:
        student_model = cast(Model, torch.compile(student_model, fullgraph=True))
        teacher_model = cast(Model, torch.compile(teacher_model, fullgraph=True))
        loss_fn = cast(Loss, torch.compile(loss_fn, fullgraph=True))

    if cfg.enable_fp32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    auto_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    save_every_epoch = cfg.save_every_epoch
    non_blocking = device.type == "cuda" and bool(dataloader.pin_memory)
    n_global_crops = cfg.n_global_crops
    clip_grad_norm_1 = cfg.clip_grad_norm_1
    ckpt_path = log_dir / f"{model_cfg.name}.pth"

    logger.info(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, n_epochs + 1):
        loss_cum = 0.0
        t_start = time()
        lr, weight_decay, momentum = 0.0, 0.0, 0.0  # placeholders
        progress_bar = tqdm(
            dataloader, dynamic_ncols=True, leave=False, disable=(not cfg.interactive), desc=f"Epoch {epoch}/{n_epochs}"
        )
        student_temp = torch.tensor(student_temp_scheduler(epoch - 1), device=device)
        teacher_temp = torch.tensor(teacher_temp_scheduler(epoch - 1), device=device)
        for step, (images, _) in enumerate(progress_bar):
            images = [img.to(device, non_blocking=non_blocking) for img in images]

            step = (epoch - 1) * batch_per_epoch + step
            # set lr and weight decay
            lr, weight_decay = lr_scheduler(step), weight_decay_scheduler(step)
            for new_group in optimizer.param_groups:
                new_group["lr"] = lr
                new_group["weight_decay"] = weight_decay if new_group["apply_decay"] else 0.0

            with auto_ctx:
                student_logits = student_model(images)
                teacher_logits = teacher_model(images[:n_global_crops])
                loss = loss_fn(student_logits, teacher_logits, student_temp, teacher_temp)

            if not math.isfinite(loss.item()):
                logger.error(f"Loss is not finite: {loss.item()}")
                return

            progress_bar.set_postfix_str(f"loss: {loss.item():.4f}")

            loss_cum += loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad_norm_1:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            if epoch <= cfg.freeze_last_layer_epoch:
                student_model.cancel_grad_last_layer()
            optimizer.step()
            momentum = momentum_scheduler(step)
            ema_update_teacher(teacher_model, student_model, momentum)

        metrics = {
            "epoch": epoch,
            "loss": loss_cum / batch_per_epoch,
            "time": time() - t_start,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "student_temp": student_temp,
            "teacher_temp": teacher_temp,
        }

        logger.info(format_metrics(metrics, epoch, n_epochs))
        wandb_logger(metrics)
        if epoch % save_every_epoch == 0 or epoch == n_epochs:
            torch.save(
                {
                    "cfg": cfg,
                    "student_model": student_model.state_dict(),
                    "teacher_model": teacher_model.state_dict(),
                    "dino_loss": loss_fn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
