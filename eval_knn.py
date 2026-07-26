"""Evaluate DINO backbone features with the upstream weighted k-NN classifier."""

from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import init_dataloader, init_dataset
from dino import Model, ModelConfig
from utils import torch_compile_ckpt_fix, torch_get_device


def parse_args() -> Namespace:
    """Parse weighted k-nearest-neighbor evaluation arguments.

    Returns:
        Parsed evaluation arguments.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_path", type=Path, nargs="?", help="Path to a DINO training checkpoint.")
    parser.add_argument(
        "--model-key",
        choices=("teacher_model", "student_model"),
        default="teacher_model",
        help="Checkpoint model to evaluate (default: teacher_model).",
    )
    parser.add_argument(
        "--nb-knn",
        default=[10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Numbers of neighbors to use (default: 10 20 100 200).",
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="Temperature for similarity-weighted voting (default: 0.07).",
    )
    parser.add_argument("--dump-features", type=Path, help="Directory in which to save extracted features and labels.")
    parser.add_argument(
        "--load-features", type=Path, help="Directory from which to load extracted features and labels."
    )
    return parser.parse_args()


def extract_features(
    model: Model, dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract unnormalized features and labels from a data loader.

    Args:
        model: DINO model whose ViT backbone encodes images.
        dataloader: Evaluation data loader yielding image and label batches.
        device: Device used for model inference.

    Returns:
        CPU feature matrix and matching label vector.
    """
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    non_blocking = device.type == "cuda" and bool(dataloader.pin_memory)
    for images, labels in tqdm(dataloader, dynamic_ncols=True, desc="Extracting features"):
        features = model.forward_features(images.to(device, non_blocking=non_blocking))
        feature_batches.append(features.detach().cpu())
        label_batches.append(labels.detach().cpu())
    return torch.cat(feature_batches), torch.cat(label_batches)


def save_features(
    feature_dir: Path,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    """Save extracted features and labels in upstream-compatible filenames.

    Args:
        feature_dir: Destination directory.
        train_features: Training feature matrix.
        test_features: Validation feature matrix.
        train_labels: Training labels.
        test_labels: Validation labels.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_features, feature_dir / "trainfeat.pth")
    torch.save(test_features, feature_dir / "testfeat.pth")
    torch.save(train_labels, feature_dir / "trainlabels.pth")
    torch.save(test_labels, feature_dir / "testlabels.pth")


def load_features(feature_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load extracted features and labels from upstream-compatible filenames.

    Args:
        feature_dir: Directory containing the saved tensors.

    Returns:
        Training features, validation features, training labels, and validation labels.
    """
    train_features = cast(torch.Tensor, torch.load(feature_dir / "trainfeat.pth", weights_only=True))
    test_features = cast(torch.Tensor, torch.load(feature_dir / "testfeat.pth", weights_only=True))
    train_labels = cast(torch.Tensor, torch.load(feature_dir / "trainlabels.pth", weights_only=True))
    test_labels = cast(torch.Tensor, torch.load(feature_dir / "testlabels.pth", weights_only=True))
    return train_features, test_features, train_labels, test_labels


def knn_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int,
    temperature: float,
    device: torch.device,
) -> tuple[float, float]:
    """Compute upstream-style similarity-weighted k-NN top-1 and top-5 accuracy.

    Args:
        train_features: Training feature matrix.
        train_labels: Training labels.
        test_features: Validation feature matrix.
        test_labels: Validation labels.
        k: Number of nearest neighbors for each vote.
        temperature: Temperature used for voting weights.
        device: Device used for classification.

    Returns:
        Top-1 and top-5 accuracy percentages.
    """
    if k < 1 or k > train_features.shape[0]:
        raise ValueError(f"k must be between 1 and {train_features.shape[0]}, got {k}.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    train_features = F.normalize(train_features.to(device), dim=1, p=2).T
    test_features = F.normalize(test_features.to(device), dim=1, p=2)
    train_labels = train_labels.to(device=device, dtype=torch.long)
    test_labels = test_labels.to(device=device, dtype=torch.long)
    num_classes = int(torch.maximum(train_labels.max(), test_labels.max()).item()) + 1
    top1_correct = 0
    top5_correct = 0
    chunk_size = max(1, test_labels.shape[0] // 100)

    for start in range(0, test_labels.shape[0], chunk_size):
        features = test_features[start : start + chunk_size]
        targets = test_labels[start : start + chunk_size]
        distances, indices = (features @ train_features).topk(k, largest=True, sorted=True)
        retrieved_labels = train_labels.expand(targets.shape[0], -1).gather(1, indices)
        votes = F.one_hot(retrieved_labels, num_classes=num_classes).to(distances.dtype)
        probabilities = (votes * (distances / temperature).exp().unsqueeze(-1)).sum(dim=1)
        predictions = probabilities.topk(min(5, num_classes), dim=1).indices
        correct = predictions.eq(targets.unsqueeze(1))
        top1_correct += int(correct[:, 0].sum().item())
        top5_correct += int(correct.any(dim=1).sum().item())

    total = test_labels.shape[0]
    return 100.0 * top1_correct / total, 100.0 * top5_correct / total


@torch.inference_mode()
def main(args: Namespace) -> None:
    """Extract or load features, then report weighted k-NN accuracy.

    Args:
        args: Checkpoint, feature cache, and classifier settings.
    """
    load_feature_dir = cast(Path | None, args.load_features)
    if load_feature_dir is not None:
        train_features, test_features, train_labels, test_labels = load_features(load_feature_dir)
    else:
        ckpt_path = cast(Path | None, args.ckpt_path)
        if ckpt_path is None:
            raise ValueError("ckpt_path is required unless --load-features is specified.")
        device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")
        checkpoint = cast(MutableMapping[str, object], torch.load(ckpt_path, map_location="cpu", weights_only=False))
        cfg = cast(DictConfig, checkpoint["cfg"])
        model_cfg = ModelConfig(**cfg.model)
        model = Model(model_cfg)
        model_key = cast(str, args.model_key)
        state_dict = cast(MutableMapping[str, torch.Tensor], checkpoint[model_key])
        model.load_state_dict(torch_compile_ckpt_fix(state_dict))
        model.to(device)
        model.eval()
        if bool(cfg.torch_compile):
            model = cast(Model, torch.compile(model, fullgraph=True))

        train_dataset = init_dataset(cfg, split="train", train_mode=False)
        train_dataloader = cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]], init_dataloader(train_dataset, cfg, train_mode=False)
        )
        test_dataset = init_dataset(cfg, split="validation", train_mode=False)
        test_dataloader = cast(
            DataLoader[tuple[torch.Tensor, torch.Tensor]], init_dataloader(test_dataset, cfg, train_mode=False)
        )
        train_features, train_labels = extract_features(model, train_dataloader, device)
        test_features, test_labels = extract_features(model, test_dataloader, device)

        dump_feature_dir = cast(Path | None, args.dump_features)
        if dump_feature_dir is not None:
            save_features(dump_feature_dir, train_features, test_features, train_labels, test_labels)

    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")
    temperature = cast(float, args.temperature)
    for k in cast(list[int], args.nb_knn):
        top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, k, temperature, device)
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")


if __name__ == "__main__":
    main(parse_args())
