# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vision tower implementation for Kimi-K2.5 model.

This module provides the vision encoder components for Kimi-K2.5,
including 3D patch embedding, RoPE position embedding, and
temporal pooling for video chunks.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import threading
from functools import lru_cache
from transformers.activations import GELUActivation

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import (
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)
from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

logger = init_logger(__name__)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope_npu(xq, xk, freqs_cis):
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    
    xq = xq.unsqueeze(0)  # (1, S, H, D)
    xk = xk.unsqueeze(0)

    cos = torch.cat([freqs_cis.real, freqs_cis.real], dim=-1).to(xq.dtype)
    sin = torch.cat([freqs_cis.imag, freqs_cis.imag], dim=-1).to(xq.dtype)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    xq_out = torch_npu.npu_rotary_mul(xq, cos, sin).squeeze(0)
    xk_out = torch_npu.npu_rotary_mul(xk, cos, sin).squeeze(0)
    return xq_out, xk_out

def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sincos positional embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """Generate 1D sincos positional embedding."""
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """2D learnable position embedding with temporal extension."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            x_device = x.device
            x_dtype = x.dtype
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                weight_fp32 = self.weight.to(dtype=torch.float32)
                weight_cpu = weight_fp32.to("cpu")
                pos_emb_2d = get_rope_shape(
                    weight_cpu,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )
                pos_emb_2d = pos_emb_2d.to(x_device, dtype=x_dtype)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out

class AscendMoonVision3dPatchEmbed(nn.Module):
    """3D patch embedding for vision tower. 昇腾NPU适配版，纯手动实现Conv2d"""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), (
            f"Invalid patch_size type: {type(patch_size)}"
        )
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, (
            f"Expected patch_size to be a tuple of 2, got {patch_size}"
        )
        self.patch_size = patch_size  # (ph, pw)
        self.out_dim = out_dim
        self.in_dim = in_dim

        # 保留原Conv2d层（仅复用权重/偏置参数，不调用其forward）
        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        # 位置编码与原代码完全一致，无修改
        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:

        ph, pw = self.patch_size  # patch高度、宽度
        B, C, H, W = x.shape      # 输入维度：批次、通道、高、宽
        nH = H // ph
        nW = W // pw

        patch_x = x.view(B, C, nH, ph, nW, pw)
        patch_x = patch_x.permute(0, 2, 4, 1, 3, 5)
        patch_x = patch_x.reshape(B, nH * nW, C * ph * pw)
        conv_weight = self.proj.weight.data.view(self.out_dim, -1)
        x_proj = patch_x.matmul(conv_weight.transpose(0, 1))


        x = x_proj.view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)

        return x

class MoonVision3dPatchEmbed(nn.Module):
    """3D patch embedding for vision tower."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), (
            f"Invalid patch_size type: {type(patch_size)}"
        )
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, (
            f"Expected patch_size to be a tuple of 2, got {patch_size}"
        )
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support."""

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self._cache_max_size = 128  # 新增：缓存最大条目数
        self._cache_access_order = []  # 新增：记录访问顺序，用于LRU

        self.register_buffer(
            "freqs_cis", 
            self._precompute_freqs_cis(torch.device('cpu')), 
            persistent=False
        )
        self._cached_cos_sin = {}
        self._lock = threading.Lock()  # 线程安全锁

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, "
            f"max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid."""
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        # if not hasattr(self, "freqs_cis"):
        #     self.register_buffer(
        #         "freqs_cis", self._precompute_freqs_cis(device), persistent=False
        #     )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis
    
    def _evict_lru_cache(self):
        while len(self._cached_cos_sin) > self._cache_max_size:
            lru_key = self._cache_access_order.pop(0)
            if lru_key in self._cached_cos_sin:
                del self._cached_cos_sin[lru_key]

    def get_cached_cos_sin_for_shapes(
        self, grid_thws: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shapes_tuple = tuple(tuple(map(int, row)) for row in grid_thws.tolist())
        cache_key = (shapes_tuple, dtype, device)
        
        with self._lock:
            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)

            if cache_key not in self._cached_cos_sin:
                freqs_cis = self.get_freqs_cis(grid_thws, device).to(torch.complex64)
                cos = torch.cat([freqs_cis.real, freqs_cis.real], dim=-1).to(dtype)
                sin = torch.cat([freqs_cis.imag, freqs_cis.imag], dim=-1).to(dtype)
                cos = cos.unsqueeze(0).unsqueeze(2)
                sin = sin.unsqueeze(0).unsqueeze(2)
                self._cached_cos_sin[cache_key] = (cos, sin)
                self._evict_lru_cache()
        
        return self._cached_cos_sin[cache_key]

class MLP2(nn.Module):
    """Two-layer MLP with tensor parallel support."""

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        assert len(dims) == 3
        self.use_data_parallel = use_data_parallel
        self.fc0 = ColumnParallelLinear(
            dims[0],
            dims[1],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc0"),
            disable_tp=self.use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            dims[1],
            dims[2],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=self.use_data_parallel,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc0(x)
        x = self.activation(x)
        x, _ = self.fc1(x)
        return x


class MoonViTEncoderLayer(nn.Module):
    """Single encoder layer for MoonViT with TP/DP support."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_attention_heads_per_partition = divide(num_heads, self.tp_size)

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=self.use_data_parallel,
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=hidden_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=attn_bias,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            hidden_dim,
            hidden_dim,
            bias=attn_bias,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        cached_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,  # 新增参数
    ):
        """Compute self-attention with packed QKV.

        Args:
            x (torch.Tensor): (seqlen, hidden_dim)
            cu_seqlens (torch.Tensor): cumulative sequence lengths
        """
        seq_length = x.size(0)
        xqkv, _ = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        # xq, xk = apply_rope(xq, xk, rope_freqs_cis)
        # 使用缓存的 cos/sin 或计算新的
        if cached_cos_sin is not None:
            cos, sin = cached_cos_sin
            xq = torch_npu.npu_rotary_mul(xq.unsqueeze(0), cos, sin).squeeze(0)
            xk = torch_npu.npu_rotary_mul(xk.unsqueeze(0), cos, sin).squeeze(0)
        else:
            xq, xk = apply_rope_npu(xq, xk, rope_freqs_cis)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_out = self.attn(
            xq.unsqueeze(0),
            xk.unsqueeze(0),
            xv.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_out = attn_out.reshape(
            seq_length,
            self.num_attention_heads_per_partition
            * self.hidden_size_per_attention_head,
        )
        attn_out, _ = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
        cached_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,  # 新增参数
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(
            hidden_states, cu_seqlens, rope_freqs_cis, cached_cos_sin=cached_cos_sin
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    """Full encoder stack for MoonViT 3D."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert video_attn_type == "spatial_temporal", (
            f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        )
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device
        )

        rope_freqs_cis = rope_freqs_cis.to(torch.complex64)
        cached_cos_sin = None
        try:
            cached_cos_sin = self.rope_2d.get_cached_cos_sin_for_shapes(
                grid_thws, hidden_states.dtype, hidden_states.device
            )
        except Exception as e:
            logger.warning(f"Cache miss/failure for cos/sin: {e}, falling back to original method")
            cached_cos_sin = None

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
                cached_cos_sin=cached_cos_sin
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    """Temporal pooling patch merger."""
    kh, kw = merge_kernel_size
    lengths = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
    seqs = x.split(lengths, dim=0)

    outputs = []
    for seq, (t, h, w) in zip(seqs, grid_thws.tolist()):
        nh, nw = h // kh, w // kw
        # Reshape: (t*h*w, d) -> (t, nh, kh, nw, kw, d)
        v = seq.view(t, nh, kh, nw, kw, -1)
        # Temporal pooling first (reduces tensor size before permute)
        v = v.mean(dim=0)  # (nh, kh, nw, kw, d)
        # Spatial rearrangement: (nh, kh, nw, kw, d) -> (nh, nw, kh, kw, d)
        out = v.permute(0, 2, 1, 3, 4).reshape(nh * nw, kh * kw, -1)
        outputs.append(out)

    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    """Main vision tower model.

    Uses KimiK25VisionConfig directly from transformers_utils/configs/kimi_k25.py.
    """

    def __init__(
        self,
        config: KimiK25VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = deepcopy(config)
        self.config = config  # Required for run_dp_sharded_mrope_vision_model
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": get_act_fn("gelu_pytorch_tanh"),
                "attn_bias": True,
            },
            video_attn_type=config.video_attn_type,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "encoder"),
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if (
            self.merge_type == "sd2_tpool"
        ):  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(
                hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
            )
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")

        return hidden_states


@torch.inference_mode()
def mm_projector_forward(mm_projector: torch.nn.Module, vt_output: list[torch.Tensor]):
    """Apply MM projector to vision tower outputs."""
    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched)
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    proj_out = torch.split(proj_out, num_embedding_list)
    return proj_out


@torch.inference_mode()
def vision_tower_forward(
    vision_tower: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: Any,
    use_data_parallel: bool,
) -> list[torch.Tensor]:
    """DP-sharded vision tower forward with mrope.

    Uses vLLM's standard data parallelism utility to shard the batch
    across available GPUs, enabling parallel processing of vision features.
    """
    if use_data_parallel:
        grid_thw_list = grid_thw.tolist()
        vt_outputs = run_dp_sharded_mrope_vision_model(
            vision_model=vision_tower,
            pixel_values=pixel_values,
            grid_thw_list=grid_thw_list,
            rope_type="rope_2d",
        )
    else:
        vt_outputs = vision_tower(pixel_values, grid_thw)
    tensors = mm_projector_forward(mm_projector, list(vt_outputs))
    return list(tensors)


class KimiK25MultiModalProjector(nn.Module):
    """Multi-modal projector with patch merging for Kimi-K2.5."""

    def __init__(
        self,
        config: KimiK25VisionConfig,
        use_data_parallel: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.hidden_size * merge_h * merge_w

        self.pre_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.mm_hidden_size,
            bias=True,
            prefix=f"{prefix}.linear_2",
        )
        self.act = GELUActivation()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states
