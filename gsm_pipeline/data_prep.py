from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from .config import ChannelConfig, FeatureConfig
from .features import FeatureExtractionResult, extract_feature_tensors


@dataclass
class ChannelArtifacts:
    """Holds the generated channel tensor and metadata."""

    channel_tensor: np.ndarray
    metadata: Dict[str, Any]


def _configure_tensorflow():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as exc:
            print(f"TensorFlow memory growth configuration failed: {exc}")
    tf.get_logger().setLevel("ERROR")


def generate_channel_tensor(cfg: ChannelConfig) -> ChannelArtifacts:
    """Generate the 3GPP channel tensor using Sionna."""

    _configure_tensorflow()

    from sionna.phy.channel.tr38901 import AntennaArray, RMa, UMa, UMi
    from sionna.phy.channel import (
        cir_to_ofdm_channel,
        gen_single_sector_topology,
        set_3gpp_scenario_parameters,
        subcarrier_frequencies,
    )

    bs_array = AntennaArray(
        num_rows=cfg.bs_array_rows,
        num_cols=cfg.bs_array_cols,
        polarization=cfg.polarization,
        polarization_type=cfg.polarization_type,
        antenna_pattern=cfg.antenna_pattern,
        carrier_frequency=cfg.carrier_frequency,
        vertical_spacing=cfg.bs_vertical_spacing,
        horizontal_spacing=cfg.bs_horizontal_spacing,
    )

    ue_array = AntennaArray(
        num_rows=cfg.ue_array_rows,
        num_cols=cfg.ue_array_cols,
        polarization=cfg.polarization,
        polarization_type=cfg.polarization_type,
        antenna_pattern=cfg.antenna_pattern,
        carrier_frequency=cfg.carrier_frequency,
        vertical_spacing=cfg.ue_vertical_spacing,
        horizontal_spacing=cfg.ue_horizontal_spacing,
    )

    scenario = cfg.scenario.lower()
    if scenario == "uma":
        channel_model = UMa(
            carrier_frequency=cfg.carrier_frequency,
            o2i_model="low",
            ut_array=ue_array,
            bs_array=bs_array,
            direction="downlink",
        )
    elif scenario == "umi":
        channel_model = UMi(
            carrier_frequency=cfg.carrier_frequency,
            o2i_model="low",
            ut_array=ue_array,
            bs_array=bs_array,
            direction="downlink",
        )
    elif scenario == "rma":
        channel_model = RMa(
            carrier_frequency=cfg.carrier_frequency,
            o2i_model="low",
            ut_array=ue_array,
            bs_array=bs_array,
            direction="downlink",
        )
    else:
        raise ValueError(f"Unsupported scenario '{cfg.scenario}'")

    (
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
    ) = set_3gpp_scenario_parameters(
        scenario=scenario,
        min_bs_ut_dist=cfg.min_bs_ut_dist,
        isd=cfg.isd,
        bs_height=cfg.bs_height,
        min_ut_height=cfg.ut_height,
        max_ut_height=cfg.ut_height,
        indoor_probability=0,
        min_ut_velocity=cfg.ue_speed,
        max_ut_velocity=cfg.ue_speed,
    )

    topology = gen_single_sector_topology(
        batch_size=cfg.batch_size,
        num_ut=cfg.num_ue,
        isd=isd,
        min_bs_ut_dist=min_bs_ut_dist,
        scenario=scenario,
        min_ut_height=min_ut_height,
        max_ut_height=max_ut_height,
        indoor_probability=indoor_probability,
        min_ut_velocity=min_ut_velocity,
        max_ut_velocity=max_ut_velocity,
    )

    ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    channel_model.set_topology(
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los=cfg.los,
    )
    if cfg.plot_topology:
        channel_model.show_topology()

    a, tau = channel_model(
        num_time_samples=cfg.num_ofdm_symbols,
        sampling_frequency=cfg.sc_spacing,
    )
    subcarrier_freqs = subcarrier_frequencies(cfg.fft_size, cfg.sc_spacing)
    H_freq = cir_to_ofdm_channel(subcarrier_freqs, a, tau, normalize=True)
    H_freq = np.transpose(np.squeeze(H_freq), (3, 2, 4, 1, 0))

    metadata = {
        "config": asdict(cfg),
        "topology": {
            "ut_loc": ut_loc.numpy(),
            "bs_loc": bs_loc.numpy(),
            "ut_orientations": ut_orientations.numpy(),
            "bs_orientations": bs_orientations.numpy(),
            "ut_velocities": ut_velocities.numpy(),
            "in_state": in_state.numpy(),
        },
    }
    return ChannelArtifacts(channel_tensor=H_freq.astype(np.complex64), metadata=metadata)


def prepare_feature_tensors(
    channel_tensor: np.ndarray, feature_cfg: FeatureConfig
) -> FeatureExtractionResult:
    """Wrapper that extracts both feature spaces from the channel tensor."""

    return extract_feature_tensors(channel_tensor, feature_cfg)

