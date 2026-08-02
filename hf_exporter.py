"""Export a DINO training checkpoint as a Hugging Face ViT backbone."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig
from PIL.Image import Resampling
from transformers import ViTConfig, ViTImageProcessor, ViTModel

from dino import ModelConfig
from utils import torch_compile_ckpt_fix


def build_hf_config(model_cfg: ModelConfig) -> ViTConfig:
    """Build a Transformers ViT configuration matching the DINO backbone.

    Args:
        model_cfg: Configuration used to construct the DINO model.

    Returns:
        ViT configuration for the exported backbone.
    """
    return ViTConfig(
        image_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size,
        num_channels=model_cfg.img_chls,
        hidden_size=model_cfg.embed_dim,
        num_hidden_layers=model_cfg.n_layer,
        num_attention_heads=model_cfg.n_heads,
        intermediate_size=int(model_cfg.embed_dim * model_cfg.mlp_ratio),
        hidden_dropout_prob=model_cfg.drop_rate,
        attention_probs_dropout_prob=model_cfg.attn_drop_rate,
        layer_norm_eps=1e-5,
    )


def build_hf_state_dict(local_state: Mapping[str, torch.Tensor], hf_cfg: ViTConfig) -> dict[str, torch.Tensor]:
    """Map DINO ViT checkpoint parameters to Transformers parameter names.

    Args:
        local_state: DINO model state dictionary, including the ``vit.`` prefix.
        hf_cfg: Transformers ViT configuration describing the backbone depth.

    Returns:
        State dictionary compatible with ``ViTModel`` without its pooler.
    """
    state = {key.removeprefix("vit."): value for key, value in local_state.items() if key.startswith("vit.")}
    hf_state: dict[str, torch.Tensor] = {}

    def take(local_key: str, hf_key: str) -> None:
        try:
            hf_state[hf_key] = state.pop(local_key)
        except KeyError as error:
            raise KeyError(f"Checkpoint is missing expected parameter '{local_key}'.") from error

    take("cls_token", "embeddings.cls_token")
    take("pos_embed", "embeddings.position_embeddings")
    take("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight")
    take("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias")

    for layer_idx in range(hf_cfg.num_hidden_layers):
        local_block = f"blocks.{layer_idx}"
        hf_block = f"layers.{layer_idx}"
        take(f"{local_block}.norm1.weight", f"{hf_block}.layernorm_before.weight")
        take(f"{local_block}.norm1.bias", f"{hf_block}.layernorm_before.bias")
        take(f"{local_block}.norm2.weight", f"{hf_block}.layernorm_after.weight")
        take(f"{local_block}.norm2.bias", f"{hf_block}.layernorm_after.bias")

        qkv_weight = state.pop(f"{local_block}.attn.in_proj_weight")
        qkv_bias = state.pop(f"{local_block}.attn.in_proj_bias")
        query_weight, key_weight, value_weight = qkv_weight.chunk(3, dim=0)
        query_bias, key_bias, value_bias = qkv_bias.chunk(3, dim=0)
        hf_state[f"{hf_block}.attention.q_proj.weight"] = query_weight
        hf_state[f"{hf_block}.attention.k_proj.weight"] = key_weight
        hf_state[f"{hf_block}.attention.v_proj.weight"] = value_weight
        hf_state[f"{hf_block}.attention.q_proj.bias"] = query_bias
        hf_state[f"{hf_block}.attention.k_proj.bias"] = key_bias
        hf_state[f"{hf_block}.attention.v_proj.bias"] = value_bias
        take(f"{local_block}.attn.out_proj.weight", f"{hf_block}.attention.o_proj.weight")
        take(f"{local_block}.attn.out_proj.bias", f"{hf_block}.attention.o_proj.bias")
        take(f"{local_block}.mlp.fc1.weight", f"{hf_block}.mlp.fc1.weight")
        take(f"{local_block}.mlp.fc1.bias", f"{hf_block}.mlp.fc1.bias")
        take(f"{local_block}.mlp.fc2.weight", f"{hf_block}.mlp.fc2.weight")
        take(f"{local_block}.mlp.fc2.bias", f"{hf_block}.mlp.fc2.bias")

    take("norm.weight", "layernorm.weight")
    take("norm.bias", "layernorm.bias")
    if state:
        unknown = ", ".join(sorted(state))
        raise ValueError(f"Checkpoint has unsupported ViT parameters: {unknown}")
    return hf_state


def main() -> None:
    """Parse arguments and export a checkpoint to Transformers format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Training .pth checkpoint written by train.py")
    parser.add_argument("output_dir", type=Path, help="Directory to write the Hugging Face model")
    parser.add_argument(
        "--model-key",
        choices=("teacher", "student"),
        default="teacher",
        help="Checkpoint model to export (default: teacher).",
    )
    args = parser.parse_args()

    checkpoint = cast(dict[str, object], torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    cfg = cast(DictConfig, checkpoint["cfg"])
    model_cfg = ModelConfig(**cfg.model)
    checkpoint_model_key = f"{args.model_key}_model"
    local_state = torch_compile_ckpt_fix(cast(dict[str, torch.Tensor], checkpoint[checkpoint_model_key]))
    hf_cfg = build_hf_config(model_cfg)
    hf_model = ViTModel(hf_cfg, add_pooling_layer=False)
    hf_model.load_state_dict(build_hf_state_dict(local_state, hf_cfg), strict=True)

    processor = ViTImageProcessor(
        do_resize=True,
        size={"height": model_cfg.img_size, "width": model_cfg.img_size},
        resample=Resampling.BICUBIC,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print(f"Exported Transformers ViT model to {args.output_dir}")


if __name__ == "__main__":
    main()
