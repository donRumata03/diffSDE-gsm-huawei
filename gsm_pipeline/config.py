from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple


@dataclass
class ChannelConfig:
    """Parameters that control Sionna channel generation."""

    num_bs: int = 1
    num_ue: int = 20
    num_ue_ant: int = 4
    batch_size: int = 40
    scenario: str = "UMa"
    min_bs_ut_dist: float = 100.0
    isd: float = 500.0
    bs_height: float = 25.0
    ut_height: float = 1.5
    ue_speed: float = 0.83
    carrier_frequency: float = 3.5e9
    fft_size: int = 256
    sc_spacing: float = 30e3
    num_ofdm_symbols: int = 1
    bs_array_rows: int = 4
    bs_array_cols: int = 8
    bs_vertical_spacing: float = 0.9
    bs_horizontal_spacing: float = 0.5
    ue_array_rows: int = 1
    ue_array_cols: int = 2
    ue_vertical_spacing: float = 1.0
    ue_horizontal_spacing: float = 1.0
    polarization: str = "dual"
    polarization_type: str = "cross"
    antenna_pattern: str = "38.901"
    los: Optional[bool] = False
    plot_topology: bool = False


@dataclass
class FeatureConfig:
    """Parameters for feature extraction in the frequency domain."""

    tau_rank_rel: float = 1e-3
    tau_degen_rel: float = 1e-4
    k_trunc: Optional[int] = None
    allow_zero_padding: bool = True


@dataclass
class TrainingConfig:
    """Shared training hyper-parameters for SB-CFM or repo-backed solvers."""

    method: str = "dsbm_repo"  # "dsbm_repo", "sb-cfm", "dsbm_placeholder"
    batch_size: int = 8
    val_split: float = 0.1
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    model_type: str = "conv1d"
    conv_kwargs: Optional[Dict[str, float]] = field(default_factory=dict)
    mlp_kwargs: Optional[Dict[str, float]] = field(default_factory=dict)
    cfm_sigma: float = 0.5
    cfm_ot_method: str = "exact"
    dsbm_sigma_min: float = 0.01
    dsbm_sigma_max: float = 1.0
    schedule: str = "const"
    device: Optional[str] = None
    tqdm: bool = True
    seed: int = 0
    # Repository-backed solver options
    bridge_solver: str = "dsbm"  # choices: dsb, dsbm, rectifiedflow
    bridge_num_steps: int = 40
    bridge_sigma: float = 1.0
    bridge_eps: float = 1e-3
    bridge_inner_iters: int = 500
    bridge_outer_iters: int = 4
    bridge_fb_sequence: Sequence[str] = ("b", "f")
    bridge_hidden_dims: Sequence[int] = (256, 256)
    bridge_activation: str = "silu"
    bridge_first_coupling: str = "ref"


@dataclass
class SamplingConfig:
    """Configuration for SB-SDE sampling."""

    num_samples: int = 100
    steps: int = 4
    sde_sigma: float = 0.0
    schedule: str = "const"
    device: Optional[str] = None


@dataclass
class ScoreEvalConfig:
    """Controls score distribution evaluation for SWD metrics."""

    t_value: float = 0.5
    batch_size: int = 32
    max_points: int = 200_000
    n_projections: int = 256
    seed_real: int = 0
    seed_gen: int = 1
