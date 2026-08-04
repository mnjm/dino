"""Render final-layer CLS attention maps per head for validation images in a DINO checkpoint."""

from argparse import ArgumentParser, Namespace
from collections.abc import Sized
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from data import init_dataset
from dino import Model, ModelConfig
from utils import torch_compile_ckpt_fix, torch_get_device

def parse_args() -> Namespace:
    """Parse attention visualization arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_path", type=Path, help="Path to a DINO training checkpoint.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of validation images to visualize (default: 10).",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs=2,
        default=[480, 480],
        metavar=("HEIGHT", "WIDTH"),
        help="Validation image resolution (default: 480 480).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("attn.png"),
        help="Output path for the attention figure (default: attn.png).",
    )
    return parser.parse_args()


@torch.no_grad()
def main(args: Namespace) -> None:
    """Render final-layer CLS attention maps for each head and validation image.

    Args:
        args: Parsed command-line arguments.
    """
    ckpt_path = cast(Path, args.ckpt_path)
    n_samples = cast(int, args.n_samples)
    res = cast(list[int], args.res)
    assert len(res) == 2 and all(isinstance(dim, int) for dim in res)
    save_path = cast(Path, args.save_path)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")

    model_cfg = ModelConfig(**ckpt["cfg"]["model"])
    patch_size = model_cfg.patch_size
    assert res[0] % patch_size == 0 and res[1] % patch_size == 0, "resolution must be divisible by the patch size"
    model = Model(model_cfg)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["teacher_model"]))
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    dataset = init_dataset(
        ckpt["cfg"],
        split="validation",
        train_mode=False,
        inp_res=(res[0], res[1]),
    )
    n_samples = min(n_samples, len(cast(Sized, dataset)))

    imgs = torch.empty(n_samples, 3, *res)
    for idx in range(n_samples):
        imgs[idx, ...] = dataset[idx][0]

    imgs = imgs.to(device)
    attn = model.get_final_layer_attn(imgs)[:, :, 0, 1:]
    n_heads = attn.shape[1]
    h_patches, w_patches = res[0] // patch_size, res[1] // patch_size
    assert h_patches * w_patches == attn.shape[-1], (
        f"Unexpected attention shape: {attn.shape=}, {h_patches=}, {w_patches=}"
    )
    attn = attn.reshape(n_samples * n_heads, 1, h_patches, w_patches)
    attn = F.interpolate(attn, size=res, mode="nearest")
    attn = attn.view(n_samples, n_heads, *res).cpu()

    imgs = imgs * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    imgs = imgs + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).cpu()

    fig, axes = plt.subplots(n_samples, n_heads + 1, figsize=(3 * (n_heads + 1), 3 * n_samples), squeeze=False)
    for idx in range(n_samples):
        axes[idx, 0].imshow(imgs[idx])
        axes[idx, 0].set_title(f"Image {idx + 1}")
        axes[idx, 0].axis("off")
        for head_idx in range(n_heads):
            axes[idx, head_idx + 1].imshow(attn[idx, head_idx], cmap="magma")
            axes[idx, head_idx + 1].set_title(f"Head {head_idx + 1}")
            axes[idx, head_idx + 1].axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main(parse_args())
