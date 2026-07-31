"""Evaluate DINO backbone features with weighted k-NN classifier."""

from argparse import ArgumentParser, Namespace
from collections.abc import Sized
from pathlib import Path
from typing import cast

import faiss
import torch
import torch.nn.functional as F
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
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./eval_embeddings"),
        help="Directory in which to save evaluation embeddings (default: ./eval_embeddings).",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the saved FAISS index, labels, and validation embeddings.",
    )
    # TODO Matplotlib plotting
    return parser.parse_args()


@torch.no_grad()
def main(args: Namespace) -> None:
    """Build or load the cached k-NN evaluation inputs.

    Args:
        args: Parsed evaluation arguments.
    """
    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")

    save_dir = cast(Path, args.save_dir)
    faiss_device = device.index or 0
    gpu_resources: faiss.StandardGpuResources | None = None

    if args.load:
        cpu_index = faiss.read_index(str(save_dir / "train.index"))
        if device.type == "cuda" and faiss_device < faiss.get_num_gpus():
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, faiss_device, cpu_index)
        else:
            index = cpu_index
        train_labels = cast(torch.Tensor, torch.load(save_dir / "train_labels.pth", weights_only=True))
        val_embeddings = cast(list[torch.Tensor], torch.load(save_dir / "val_embeddings.pth", weights_only=True))
        val_labels = cast(list[torch.Tensor], torch.load(save_dir / "val_labels.pth", weights_only=True))
    else:
        index, train_labels, val_embeddings, val_labels = generate_index_and_embeddings(args.ckpt_path, save_dir)

    nb_knn = cast(list[int], args.nb_knn)
    temperature = cast(float, args.temperature)
    max_k = max(nb_knn)
    num_classes = int(train_labels.max().item()) + 1
    top_1_correct_by_k = {k: 0 for k in nb_knn}
    top_5_correct_by_k = {k: 0 for k in nb_knn}
    num_samples = 0
    for step_i, emb_batch in enumerate(val_embeddings):
        B = emb_batch.shape[0]
        similarities, neighbor_indices = index.search(emb_batch.numpy(), max_k)
        similarities = torch.from_numpy(similarities)  # (B, max_k)
        neighbor_indices = torch.from_numpy(neighbor_indices)  # (B, max_k)
        neighbor_labels = train_labels[neighbor_indices]  # (B, max_k)
        batch_labels = val_labels[step_i]  # (B)
        weights = similarities.div_(temperature).exp_()  # e^{similarities / temperature}

        for k in nb_knn:
            class_scores = torch.zeros((B, num_classes), dtype=weights.dtype)
            # Add each neighbor's weight to its corresponding class-score column.
            class_scores.scatter_add_(1, neighbor_labels[:, :k], weights[:, :k])
            predictions = class_scores.topk(min(5, num_classes), dim=1).indices
            correct = predictions.eq(batch_labels.unsqueeze(1))
            top_1_correct_by_k[k] += int(correct[:, 0].sum().item())
            top_5_correct_by_k[k] += int(correct.any(dim=1).sum().item())

        num_samples += B

    for k in nb_knn:
        top_1_accuracy = top_1_correct_by_k[k] / num_samples
        top_5_accuracy = top_5_correct_by_k[k] / num_samples
        print(f"k={k}: top-1 accuracy={top_1_accuracy:.2%}, top-5 accuracy={top_5_accuracy:.2%}")


def generate_index_and_embeddings(ckpt_path: Path | None, save_dir: Path):
    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")
    ckpt_path = cast(Path | None, args.ckpt_path)
    assert ckpt_path is not None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model_cfg = ModelConfig(**cfg.model)
    model = Model(model_cfg)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt[args.model_key]))
    model.eval()
    model.to(device)
    if cfg.torch_compile and device.type == "cuda":
        model = cast(Model, torch.compile(model, fullgraph=True))

    # Build the FAISS index on the train set.
    faiss_device = device.index or 0
    gpu_resources: faiss.StandardGpuResources | None = None
    cpu_index = faiss.IndexFlatIP(model_cfg.embed_dim)
    if device.type == "cuda" and faiss_device < faiss.get_num_gpus():
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, faiss_device, cpu_index)
    else:
        index = cpu_index
    train_dataset = init_dataset(cfg, split="train", train_mode=False)
    train_dataloader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]], init_dataloader(train_dataset, cfg, train_mode=False)
    )
    batch_size = cast(int, train_dataloader.batch_size)
    pin_memory = device.type == "cuda" and train_dataloader.pin_memory
    progress_bar = tqdm(train_dataloader, desc="Building train index", dynamic_ncols=True, leave=False)
    train_labels = torch.empty(len(cast(Sized, train_dataset)), dtype=torch.long, device="cpu")
    for step, (imgs, lbls) in enumerate(progress_bar):
        imgs = imgs.to(device, non_blocking=pin_memory)
        features = F.normalize(model.forward_features(imgs), dim=-1, p=2)
        index.add(features.cpu().numpy())
        train_labels[step * batch_size : (step + 1) * batch_size] = lbls.cpu()

    # Extract validation embeddings and labels.
    val_dataset = init_dataset(cfg, split="validation", train_mode=False)
    val_dataloader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]], init_dataloader(val_dataset, cfg, train_mode=False)
    )
    pin_memory = device.type == "cuda" and val_dataloader.pin_memory
    progress_bar = tqdm(val_dataloader, desc="Evaluating", dynamic_ncols=True, leave=False)
    val_embeddings: list[torch.Tensor] = []
    val_labels: list[torch.Tensor] = []
    for imgs, lbls in progress_bar:
        imgs = imgs.to(device, non_blocking=pin_memory)
        features = model.forward_features(imgs)
        features = F.normalize(features, dim=-1, p=2)
        val_embeddings.append(features.cpu())
        val_labels.append(lbls.cpu())

    save_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(
        faiss.index_gpu_to_cpu(index) if gpu_resources is not None else index,
        str(save_dir / "train.index"),
    )
    torch.save(train_labels, save_dir / "train_labels.pth")
    torch.save(val_embeddings, save_dir / "val_embeddings.pth")
    torch.save(val_labels, save_dir / "val_labels.pth")

    return index, train_labels, val_embeddings, val_labels


if __name__ == "__main__":
    args = parse_args()
    main(args)
