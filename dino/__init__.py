"""
DINO model implementation in pytorch.

- Model
- Projection Head
- Loss
"""

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .vit import ViT


@dataclass
class ModelConfig:
    """Configuration values used to construct a DINO model.

    Attributes:
        name: Human-readable model name.
        img_size: Reference image height and width for positional embeddings.
        patch_size: Square ViT patch height and width in pixels.
        embed_dim: ViT token embedding dimension.
        n_layer: Number of ViT transformer blocks.
        n_heads: Number of attention heads per ViT block.
        mlp_ratio: Multiplier for the ViT MLP hidden dimension.
        drop_rate: ViT embedding, projection, and MLP dropout rate.
        attn_drop_rate: ViT attention-weight dropout rate.
        drop_path_rate: Maximum ViT stochastic-depth rate.
        proj_head_hidden_dim: Hidden dimension of the projection MLP.
        proj_head_bottleneck_dim: Normalized projection feature dimension.
        out_dim: Number of DINO prototype logits.
        img_chls: Number of input image channels.
    """

    name: str

    img_size: int
    patch_size: int
    embed_dim: int
    n_layer: int
    n_heads: int
    mlp_ratio: float

    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float

    proj_head_hidden_dim: int
    proj_head_bottleneck_dim: int
    out_dim: int

    img_chls: int = 3


class Model(nn.Module):
    """DINO model composed of a ViT backbone and projection head."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Initialize the DINO model.

        Args:
            cfg: Model configuration values.
        """
        super().__init__()
        self.vit = ViT(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            img_chls=cfg.img_chls,
            embed_dim=cfg.embed_dim,
            n_layer=cfg.n_layer,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            n_class=0,
        )
        self.proj_head = ProjectionHead(
            in_dim=cfg.embed_dim,
            out_dim=cfg.out_dim,
            hidden_dim=cfg.proj_head_hidden_dim,
            bottleneck_dim=cfg.proj_head_bottleneck_dim,
        )

    def forward_features(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Encode an image batch or ordered multi-crop batches with the ViT backbone.

        Args:
            x: Image batch with shape ``(B, C, H, W)`` or crop batches
                contiguously grouped by resolution.

        Returns:
            torch.Tensor: CLS features with shape ``(B, embed_dim)`` for one batch or
                ``(sum(crop_batch_sizes), embed_dim)`` for crop batches.
        """
        if not isinstance(x, list):
            return self.vit(x)

        start_idx = 0
        features: list[torch.Tensor] = []
        for end_idx in range(1, len(x) + 1):
            if end_idx == len(x) or x[end_idx].shape[-1] != x[start_idx].shape[-1]:
                features.append(self.vit(torch.cat(x[start_idx:end_idx])))
                start_idx = end_idx
        return torch.cat(features)

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Run image batches or ordered multi-crop batches through the DINO model.

        Args:
            x: Image batch with shape ``(B, C, H, W)`` or crop batches
                contiguously grouped by resolution.

        Returns:
            torch.Tensor: Projection logits with shape ``(B, out_dim)`` for one batch or
                ``(sum(crop_batch_sizes), out_dim)`` for crop batches.
        """
        return self.proj_head(self.forward_features(x))

    def cancel_grad_last_layer(self) -> None:
        """Clear projection-head final-layer gradients before an optimizer step."""
        self.proj_head.cancel_grad_last_layer()

    def params_breakdown(self) -> str:
        """Summarize total and trainable parameters by model component.

        Returns:
            str: Full-model, ViT, and head parameter counts, in millions
        """
        params_count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
        params_count_grad = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
        params = [
            ("Total", params_count(self), params_count_grad(self)),
            ("ViT", params_count(self.vit), params_count_grad(self.vit)),
            ("Head", params_count(self.proj_head), params_count_grad(self.proj_head)),
        ]
        return " | ".join(f"{name}: {total:.2f}M trainable: {trainable:.2f}M" for name, total, trainable in params)


class ProjectionHead(nn.Module):
    """DINO projection head"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        """Initialize the projection MLP.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output prototype dimension.
            hidden_dim: Hidden layer dimension.
            bottleneck_dim: Bottleneck feature dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)

        # nn.utils.parametrizations.weight_norm decouples the weight (W) into magnitude and direction g and v
        # W = g . v / ||v||
        # g = (out_dim, 1) and v = (out_dim, in_dim)
        # This is used in dino to learn prototype representations
        self.last_layer = weight_norm(  # this has to be named `last layer`
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )  # weight.original0 is g and weight.original1 is v

        # Freeze magnitude (g) to 1.0
        # Only prototypes (weight_v) are learned, combined with l2 normed embeddings (check forward)
        # logits become approx. cosine similarities between 256d embeddings and prototypes
        weight_g = self.last_layer.get_parameter("parametrizations.weight.original0")
        nn.init.constant_(weight_g, 1.0)
        weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize linear layers with DINO-style weights.

        Args:
            m: Module to initialize when it is a linear layer.
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize features.

        Args:
            x: Input features with shape ``(..., in_dim)``.

        Returns:
            torch.Tensor: Prototype logits with shape ``(..., out_dim)``.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # l2 norm
        x = self.last_layer(x)
        return x

    def cancel_grad_last_layer(self) -> None:
        """Clear gradients for all final projection-layer parameters."""
        for param in self.last_layer.parameters():
            param.grad = None


class Loss(nn.Module):
    """DINO cross-view loss with teacher centering and sharpening."""

    def __init__(
        self,
        out_dim: int,
        center_momentum: float,
        n_local_crops: int,
        n_global_crops: int,
    ) -> None:
        """Initialize schedules and the teacher-output center buffer.

        Args:
            out_dim: Output prototype dimension.
            center_momentum: EMA momentum for the teacher center.
            n_local_crops: Number of local student crops.
            n_global_crops: Number of global teacher/student crops.
        """
        super().__init__()
        self.out_dim = out_dim
        self.center_momentum = center_momentum
        self.n_total_crops = n_local_crops + n_global_crops
        self.n_global_crops = n_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center: torch.Tensor

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        tau_s: torch.Tensor,
        tau_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DINO loss across non-matching crop pairs.

        Args:
            student_logits: Concatenated student outputs for all crops.
            teacher_logits: Concatenated teacher outputs for global crops.
            tau_s: Current student temperature.
            tau_t: Current teacher temperature.

        Returns:
            torch.Tensor: Mean DINO loss for the batch.
        """
        student_logits = student_logits / tau_s
        student_logits_chunks = student_logits.chunk(self.n_total_crops)

        # Teacher centering and sharpening
        teacher_probs = F.softmax((teacher_logits - self.center) / tau_t, dim=-1)
        teacher_probs = teacher_probs.detach()  # stop gradient
        teacher_probs_chunks = teacher_probs.chunk(self.n_global_crops)

        loss_cum = teacher_logits.new_tensor(0.0)
        loss_terms = 0
        for s_crop_idx, s_crop_logits in enumerate(student_logits_chunks):
            for t_crop_idx, t_crop_probs in enumerate(teacher_probs_chunks):
                if s_crop_idx == t_crop_idx:
                    continue
                loss_cum += torch.sum(-t_crop_probs * F.log_softmax(s_crop_logits, dim=-1), dim=-1).mean()
                loss_terms += 1

        self.update_center(teacher_logits)
        return loss_cum / loss_terms

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        """Update the teacher center with the current batch mean.

        Args:
            teacher_logits: Concatenated teacher outputs for global crops.
        """
        batch_center = torch.mean(teacher_logits, keepdim=True, dim=0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


@torch.no_grad()
def ema_update_teacher(teacher: Model, student: Model, alpha: float) -> None:
    """Update teacher parameters as an exponential moving average of student parameters.

    Args:
        teacher: Model updated in place.
        student: Model providing the current parameter values.
        alpha: Weight assigned to each existing teacher parameter.
    """
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_((1 - alpha) * student_param.data)


if __name__ == "__main__":
    config = OmegaConf.load("./config/model/vit-s-16.yaml")
    model_cfg = ModelConfig(
        name=str(config.name),
        img_size=int(config.img_size),
        patch_size=int(config.patch_size),
        embed_dim=int(config.embed_dim),
        n_layer=int(config.n_layer),
        n_heads=int(config.n_heads),
        mlp_ratio=float(config.mlp_ratio),
        drop_rate=float(config.drop_rate),
        attn_drop_rate=float(config.attn_drop_rate),
        drop_path_rate=float(config.drop_path_rate),
        proj_head_hidden_dim=int(config.proj_head_hidden_dim),
        proj_head_bottleneck_dim=int(config.proj_head_bottleneck_dim),
        out_dim=65536,
        img_chls=int(config.img_chls),
    )
    model = Model(model_cfg)
    print(f"{model_cfg.name}\n{model.params_breakdown()}")
    img = torch.randn(1, 3, 224, 224)
    output = model(img)
    print(output.shape)
    img = torch.randn(1, 3, 94, 94)
    output = model(img)
    print(output.shape)
