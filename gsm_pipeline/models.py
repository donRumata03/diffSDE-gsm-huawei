from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

DEFAULT_CONV_KWARGS = {"base_dim": 128, "num_layers": 4, "kernel_size": 5}
DEFAULT_MLP_KWARGS = {"hidden_dim": 512, "num_layers": 3}


class TimeCondConvNet1D(nn.Module):
    def __init__(self, in_channels: int, base_dim: int = 128, num_layers: int = 4, kernel_size: int = 5):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.kernel_size = kernel_size
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim),
            nn.SiLU(),
        )

        conv_blocks = []
        for layer_idx in range(num_layers):
            in_dim = base_dim if layer_idx > 0 else in_channels + base_dim
            conv_blocks.append(
                nn.Conv1d(
                    in_dim,
                    base_dim,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_blocks.append(nn.GroupNorm(8, base_dim))
            conv_blocks.append(nn.SiLU())
        conv_blocks.append(
            nn.Conv1d(
                base_dim,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.net = nn.Sequential(*conv_blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        if t.ndim == 1:
            t = t[:, None]
        t_embed = self.time_embed(t).unsqueeze(-1).expand(-1, -1, L)
        x_in = torch.cat([x, t_embed], dim=1)
        return self.net(x_in)


class TimeCondMLP(nn.Module):
    def __init__(self, in_channels: int, length: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.in_channels = in_channels
        self.length = length
        self.hidden_dim = hidden_dim
        input_dim = in_channels * length

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        layers = []
        current_dim = input_dim + hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x_flat = x.view(B, C * L)
        if t.ndim == 1:
            t = t[:, None]
        t_embed = self.time_embed(t)
        x_in = torch.cat([x_flat, t_embed], dim=1)
        out = self.net(x_in)
        return out.view(B, C, L)


def build_model(
    model_type: str,
    in_channels: int,
    length: int,
    conv_kwargs: Optional[Dict[str, float]] = None,
    mlp_kwargs: Optional[Dict[str, float]] = None,
) -> nn.Module:
    model_type = model_type.lower()
    conv_params = DEFAULT_CONV_KWARGS.copy()
    if conv_kwargs:
        conv_params.update(conv_kwargs)
    mlp_params = DEFAULT_MLP_KWARGS.copy()
    if mlp_kwargs:
        mlp_params.update(mlp_kwargs)
    if model_type == "conv1d":
        return TimeCondConvNet1D(in_channels, **conv_params)
    if model_type == "mlp":
        return TimeCondMLP(in_channels, length, **mlp_params)
    raise ValueError("Unsupported model_type. Use 'conv1d' or 'mlp'.")

