"""Dataset loading, DINO multi-crop augmentation, and batch visualization utilities."""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as T
from datasets import Dataset as HFDataset
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import Image

warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module=r"PIL\.TiffImagePlugin",
)


class HFDatasetWrapper(Dataset[tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]]):
    """Wrap image-and-label Hugging Face data as a PyTorch dataset."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: Callable[[Image], list[torch.Tensor] | torch.Tensor] | None = None,
    ) -> None:
        """Initialize the dataset wrapper.

        Args:
            hf_dataset: Dataset containing ``image`` and ``label`` fields.
            transform: Optional transform applied to each image.
        """
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of examples in the wrapped dataset.

        Returns:
            int: Total dataset size.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]:
        """Load and transform a single example.

        Args:
            idx: Example index.

        Returns:
            tuple: Transformed image or crops, plus matching class label tensor.
        """
        item = self.dataset[idx]
        raw_img = cast(PILImage, item["image"])
        img = Image(T.functional.pil_to_tensor(raw_img.convert("RGB")))
        lbl = int(cast(int, item["label"]))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(lbl, dtype=torch.long)
        if isinstance(img, list | tuple):
            label = label.expand(len(img))

        return img, label


class DINOTransforms:
    """Build the global and local multi-crop augmentations used by DINO."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize global and local crop transforms.

        Args:
            cfg: Configuration with crop scales and local crop count.
        """
        global_crop_scale = tuple(cfg.global_crop_scale)
        local_crop_scale = tuple(cfg.local_crop_scale)
        self.n_local_crops = cfg.n_local_crops

        # taken from https://github.com/facebookresearch/dino/blob/main/main_dino.py#L419
        flip_jitter_trfm = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize_trfm = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.global_trfm_l = [
            T.Compose(
                [
                    T.RandomResizedCrop(224, scale=global_crop_scale, interpolation=T.InterpolationMode.BICUBIC),
                    flip_jitter_trfm,
                    T.GaussianBlur(kernel_size=23, sigma=(0.5, 2.0)),
                    normalize_trfm,
                ]
            ),
            T.Compose(
                [
                    T.RandomResizedCrop(224, scale=global_crop_scale, interpolation=T.InterpolationMode.BICUBIC),
                    flip_jitter_trfm,
                    T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.5, 2.0))], p=0.1),
                    T.RandomSolarize(threshold=0.5, p=0.5),
                    normalize_trfm,
                ]
            ),
        ]
        assert cfg.n_global_crops == len(self.global_trfm_l), f"cfg.n_global_crops should be {len(self.global_trfm_l)}"
        self.local_trfm = T.Compose(
            [
                T.RandomResizedCrop(96, scale=local_crop_scale, interpolation=T.InterpolationMode.BICUBIC),
                flip_jitter_trfm,
                T.RandomApply([T.GaussianBlur(kernel_size=11, sigma=(0.5, 2.0))], p=0.5),
                normalize_trfm,
            ]
        )

    def __call__(self, img: Image) -> list[torch.Tensor]:
        """Apply DINO multi-crop transforms to an image.

        Args:
            img: Input image tensor.

        Returns:
            list[torch.Tensor]: Two global crops followed by local crops.
        """
        return [tfms(img) for tfms in self.global_trfm_l] + [self.local_trfm(img) for _ in range(self.n_local_crops)]


def init_dataset(
    cfg: DictConfig, split: str = "train", train_mode: bool = True
) -> Dataset[tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]]:
    """Initialize a wrapped Hugging Face dataset.

    Args:
        cfg: Configuration containing dataset and transform settings.
        split: Dataset split to load; ``train`` or ``validation``.
        train_mode: Whether to apply DINO multi-crop training transforms.

    Returns:
        Dataset: Wrapped dataset that yields images or crops and labels.
    """
    dataset_name = str(OmegaConf.select(cfg, "dataset"))
    cache_dir = Path("./dataset") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    assert split in ("train", "validation")
    dataset = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))
    if not isinstance(dataset, HFDataset):
        raise TypeError(f"Expected a map-style dataset, got {type(dataset).__name__}")
    transforms = (
        DINOTransforms(cfg)
        if train_mode
        else T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    )
    return HFDatasetWrapper(dataset, transforms)


def init_dataloader(
    dataset: Dataset[tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]],
    cfg: DictConfig,
    train_mode: bool = True,
) -> DataLoader[tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]]:
    """Create a data loader from the configured loader options.

    Args:
        dataset: Dataset to iterate over.
        cfg: Configuration containing ``dataloader`` options.
        train_mode: Whether the loader is used for training, enabling shuffling and configured ``drop_last``.

    Returns:
        DataLoader: Configured loader. ``drop_last`` is enabled only for training.
    """
    loader_cfg = cfg.dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=int(loader_cfg.batch_size),
        shuffle=train_mode,
        num_workers=int(loader_cfg.num_workers),
        pin_memory=bool(loader_cfg.pin_memory),
        drop_last=bool(loader_cfg.drop_last) and train_mode,
    )


def plot_batch(
    batch: tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor],
    n_samples: int = 4,
    save_path: str | Path | None = None,
) -> None:
    """Display or save a grid of samples from a DINO batch.

    Args:
        batch: Batched crops and their class labels.
        n_samples: Maximum number of samples to show.
        save_path: Optional destination for the rendered image. Displays it when omitted.
    """
    crops, labels = batch
    crop_batches = [crops] if isinstance(crops, torch.Tensor) else list(crops)
    labels = cast(torch.Tensor, labels)

    batch_size = crop_batches[0].shape[0]
    n_samples = min(n_samples, batch_size)

    fig, axes = plt.subplots(
        n_samples,
        len(crop_batches),
        figsize=(3 * len(crop_batches), 3 * n_samples),
        squeeze=False,
    )

    for sample_idx in range(n_samples):
        for crop_idx, crop_batch in enumerate(crop_batches):
            axis = axes[sample_idx][crop_idx]
            img = crop_batch[sample_idx]
            img = ((img - img.min()) / (img.max() - img.min() + 1e-6)).permute(1, 2, 0).cpu().numpy()
            label = labels[sample_idx, crop_idx] if labels.ndim == 2 else labels[sample_idx]
            axis.imshow(img)
            axis.set_title(f"sample {sample_idx}, crop {crop_idx}, label {label.item()}")
            axis.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    """Download the configured dataset and display one DINO multi-crop sample."""
    cfg = cast(DictConfig, OmegaConf.load("config/default.yaml"))
    dataset = init_dataset(cfg, train_mode=True)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    plot_batch(batch, save_path="batch.png")
