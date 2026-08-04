"""
Vision Transformer (ViT) model implemention in pytorch.

Note: with pre-norm layer normalization as mentioned in DINO paper.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """Image to patch embedding module."""

    def __init__(
        self,
        patch_size: int,
        in_dim: int,
        out_dim: int,
        norm_lyr: type[nn.Module] | None = nn.Identity,
    ) -> None:
        """Initialize the patch embedding projection.

        Args:
            patch_size: Square patch height and width.
            in_dim: Number of input channels.
            out_dim: Output token embedding dimension.
            norm_lyr: Optional normalization layer factory.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        norm_factory = nn.Identity if norm_lyr is None else norm_lyr
        self.norm = norm_factory(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project an image batch into patch tokens.

        Args:
            x: Input images with shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Patch embeddings with shape ``(B, num_patches, out_dim)``.
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    """Transformer MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_fn: type[nn.Module] = nn.GELU,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize the two-layer feed-forward block.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden layer dimension.
            out_features: Output feature dimension.
            act_fn: Activation module factory.
            drop_rate: Dropout rate applied after each linear layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block to token embeddings.

        Args:
            x: Token embeddings.

        Returns:
            torch.Tensor: Transformed token embeddings.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Apply per-sample stochastic depth to residual branches during training."""

    def __init__(self, drop_rate: float) -> None:
        """Initialize the stochastic depth module.

        Args:
            drop_rate: Probability of dropping a residual branch.
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop residual branches during training.

        Args:
            x: Residual branch tensor.

        Returns:
            torch.Tensor: Input tensor after stochastic depth masking.
        """
        drop_rate = self.drop_rate
        if drop_rate == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random > drop_rate).to(dtype=x.dtype)
        out = x * mask / (1.0 - drop_rate)
        return out


class ViTBlock(nn.Module):
    """Transformer encoder block with attention and MLP sublayers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_dim: int,
        attn_drop_rate: float,
        proj_drop_rate: float,
        drop_path_rate: float,
        mlp_drop_rate: float | None = None,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        """Initialize the encoder block.

        Args:
            dim: Token embedding dimension.
            n_heads: Number of attention heads.
            mlp_dim: Hidden dimension in the MLP block.
            attn_drop_rate: Dropout rate inside attention.
            proj_drop_rate: Dropout rate after attention projection.
            drop_path_rate: Stochastic-depth drop rate.
            mlp_drop_rate: Optional MLP dropout override.
            act_fn: Activation module factory.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=attn_drop_rate,
            batch_first=True,
        )
        self.attn_proj_drop = nn.Dropout(proj_drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        mlp_drop_rate = proj_drop_rate if mlp_drop_rate is None else mlp_drop_rate
        self.mlp = MLP(dim, mlp_dim, dim, act_fn=act_fn, drop_rate=mlp_drop_rate)
        self.drop_path_attn = DropPath(drop_path_rate)
        self.drop_path_mlp = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer block over the token sequence.

        Args:
            x: Token embeddings with shape ``(B, N, dim)``.

        Returns:
            torch.Tensor: Updated token embeddings.
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        attn_out = self.attn_proj_drop(attn_out)
        x = x + self.drop_path_attn(attn_out)  # residual with stocastic drop path regularizer
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x)))  # residual with stocastic drop path regularizer
        return x

    def get_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Return this block's per-head self-attention weights.

        Args:
            x: Token embeddings with shape ``(B, N, dim)``.

        Returns:
            torch.Tensor: Attention weights with shape ``(B, n_heads, N, N)``.
        """
        norm_x = self.norm1(x)
        _, attn = self.attn(
            norm_x,
            norm_x,
            norm_x,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn


class ViT(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        img_chls: int,
        embed_dim: int,
        n_layer: int,
        n_heads: int,
        mlp_ratio: float,
        n_class: int = 0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Initialize the ViT backbone and classification head.

        Args:
            img_size: Square image height and width.
            patch_size: Square patch height and width.
            img_chls: Number of input image channels.
            embed_dim: Token embedding dimension.
            n_layer: Number of transformer encoder blocks.
            n_heads: Number of attention heads.
            mlp_ratio: Multiplier used to compute MLP hidden dimension from embed_dim.
            n_class: Number of output classes.
            drop_rate: Dropout rate after embeddings, attention projection, and MLP layers.
            attn_drop_rate: Dropout rate inside attention.
            drop_path_rate: Maximum stochastic-depth drop-path rate.
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be evenly divisible by patch_size")
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        mlp_dim = int(mlp_ratio * embed_dim)
        self.patch_embed = PatchEmbed(patch_size, img_chls, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList(
            ViTBlock(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=drop_rate,
                drop_path_rate=(layer_idx + 1) / n_layer * drop_path_rate,
            )
            for layer_idx in range(n_layer)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_class) if n_class > 0 else None

        # Init weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize supported module weights in place.

        Args:
            m: Module to initialize when supported.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def interpolate_pos_embed(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Interpolate positional embeddings to match a patch-grid size.

        Args:
            x: Token sequence, including its CLS token, whose shape provides the
                target token count.
            H: Target image height in pixels.
            W: Target image width in pixels.

        Returns:
            torch.Tensor: CLS and patch positional embeddings for the target grid.
        """
        _, n_tgt_patches, embed_dim = x.shape
        n_tgt_patches = n_tgt_patches - 1  # remove cls token
        n_patches = self.pos_embed.shape[1] - 1  # remove cls token
        if n_tgt_patches == n_patches:
            return self.pos_embed
        grid_size = int(math.sqrt(n_patches))
        assert n_patches == grid_size**2, f"n_patches ({n_patches}) must be equal to grid_size^2 ({grid_size**2})"
        tgt_grid_h, tgt_grid_w = H // self.patch_size, W // self.patch_size
        pos_embed = self.pos_embed[:, 1:, :]  # remove cls token  (1, n_patches, embed_dim)
        pos_embed = pos_embed.reshape(1, grid_size, grid_size, embed_dim)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, embed_dim, grid_size, grid_size)
        pos_embed = F.interpolate(
            pos_embed, size=(tgt_grid_h, tgt_grid_w), mode="bicubic", align_corners=False
        )  # (1, embed_dim, tgt_grid_h, tgt_grid_w)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)  # (1, tgt_grid_h * tgt_grid_w, embed_dim)
        assert pos_embed.shape[1] == n_tgt_patches
        return torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)  # add cls token back

    def _prepare_tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        """Embed images and add CLS and positional tokens."""
        B, _, H, W = imgs.shape
        x = self.patch_embed(imgs)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # prepend [CLS]
        x = x + self.interpolate_pos_embed(x, H, W)
        return self.pos_drop(x)

    def get_final_layer_attn(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return per-head self-attention weights from the final transformer block.

        Args:
            imgs: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Attention weights with shape
                ``(B, n_heads, 1 + num_patches, 1 + num_patches)``.
        """
        x = self._prepare_tokens(imgs)
        for blk in self.blocks[:-1]:
            x = blk(x)
        final_block = self.blocks[-1]
        assert isinstance(final_block, ViTBlock)
        return final_block.get_attn(x)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run the model over image batches.

        Args:
            imgs: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Classification logits with shape ``(B, n_class)`` when a
                head exists; otherwise CLS embeddings with shape ``(B, embed_dim)``.
        """
        x = self._prepare_tokens(imgs)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # take [CLS]

        return self.head(cls_out) if self.head is not None else cls_out
