"""Train and evaluate a linear classifier on frozen DINO features."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import init_dataset
from dino import Model, ModelConfig
from utils import torch_compile_ckpt_fix, torch_get_device, torch_set_seed

torch_set_seed(42)

n_epochs = 20
batch_size = 256
lr = 1e-3 * batch_size / 256
n_workers = 8
weight_decay = 0
val_epoch_freq = 1
n_last_blocks = 4 # use the last 4 transformer blocks's CLS token features

class LinearClassifier(nn.Module):
    """Linear layer trained on top of frozen backbone features."""

    def __init__(self, feature_dim: int, num_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_labels)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a batch of backbone features."""
        return self.linear(features.flatten(start_dim=1))


def parse_args() -> Namespace:
    """Parse the checkpoint and backbone selection arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_path", type=Path, help="Path to a DINO training checkpoint.")
    parser.add_argument(
        "--model-key",
        choices=("teacher_model", "student_model"),
        default="teacher_model",
        help="Checkpoint backbone to evaluate (default: teacher_model).",
    )
    return parser.parse_args()

def build_dataloader(cfg: DictConfig, split: str, train_mode: bool) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a linear-evaluation loader using the checkpoint's dataset settings."""
    dataset = init_dataset(cfg, split=split, train_mode=False)
    return cast(DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
    ))


def extract_features(model: Model, images: torch.Tensor) -> torch.Tensor:
    """Concatenate CLS features from the final four transformer blocks."""
    with torch.no_grad():
        intermediate_outputs = model.get_intermediate_layers(images, n=n_last_blocks)
        return torch.cat([output[:, 0] for output in intermediate_outputs], dim=-1)


def train_epoch(
    model: Model,
    clfr_head: LinearClassifier,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epoch: int,
) -> float:
    """Train the classifier for one epoch and return mean cross-entropy loss."""
    clfr_head.train()
    loss_total = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}: Train", dynamic_ncols=True, leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=loader.pin_memory)
        labels = labels.to(device, non_blocking=loader.pin_memory)
        logits = clfr_head(extract_features(model, images))
        loss = nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        progress_bar.set_postfix_str(f"loss={loss.item():.4f}")
    return loss_total / len(loader)


@torch.no_grad()
def val_epoch(
    model: Model,
    clfr_head: LinearClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_labels: int,
    device: torch.device,
    epoch
) -> tuple[float, float, float]:
    """Return validation loss and top-1/top-5 accuracy."""
    clfr_head.eval()
    loss_total = 0.0
    top_1_correct = 0
    top_5_correct = 0
    num_samples = 0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs} Val", dynamic_ncols=True, leave=False):
        images = images.to(device, non_blocking=loader.pin_memory)
        labels = labels.to(device, non_blocking=loader.pin_memory)
        logits = clfr_head(extract_features(model, images))
        loss_total += nn.functional.cross_entropy(logits, labels).item()
        predictions = logits.topk(min(5, num_labels), dim=1).indices
        correct = predictions.eq(labels.unsqueeze(1))
        top_1_correct += int(correct[:, 0].sum().item())
        top_5_correct += int(correct.any(dim=1).sum().item())
        num_samples += labels.numel()
    return loss_total / len(loader), top_1_correct / num_samples, top_5_correct / num_samples


def main(args: Namespace) -> None:
    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["cfg"]
    model_cfg = ModelConfig(**ckpt_cfg.model)
    model = Model(model_cfg)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt[args.model_key]))
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    train_loader = build_dataloader(ckpt_cfg, split="train", train_mode=True)
    val_loader = build_dataloader(ckpt_cfg, split="validation", train_mode=False)

    assert ckpt_cfg['dataset'] == "ethz/food101", "Only food101 dataset is supported"
    num_labels = 101

    clfr_head = LinearClassifier(n_last_blocks * model_cfg.embed_dim, num_labels).to(device)
    optimizer = torch.optim.SGD(
        clfr_head.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    best_acc = 0.0

    print(f"Model: {model_cfg.name} dataset: {ckpt_cfg['dataset']}")
    val_loss, top_1_acc, top_5_acc = val_epoch(model, clfr_head, val_loader, num_labels, device, 0)
    print(f"Initial val_loss={val_loss:.4f} top-1-acc={top_1_acc:.2%} top-5-acc={top_5_acc:.2%}")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, clfr_head, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        if epoch % val_epoch_freq == 0 or epoch == n_epochs:
            val_loss, top_1_acc, top_5_acc = val_epoch(model, clfr_head, val_loader, num_labels, device, epoch)
            best_acc = max(best_acc, top_1_acc)
            print(f"Epoch {epoch}/{n_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} top-1-acc={top_1_acc:.2%} top-5-acc={top_5_acc:.2%}")

    print(f"Best top-1 accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main(parse_args())
