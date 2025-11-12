from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import FeatureConfig

ArrayLike = Union[np.ndarray, torch.Tensor]
_SVD_MAX_RETRIES = 3


@dataclass
class FeatureExtractionResult:
    """Container for the two feature spaces we operate with."""

    amp_phase_features: np.ndarray
    amp_phase_metadata: Dict[str, Any]
    svd_features: np.ndarray
    svd_metadata: Dict[str, Any]
    svd_summary: Dict[str, Any]


def amplitude_phase_normalization(H: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    amplitude = np.abs(H)
    phase = np.angle(H)

    amp_mean = float(amplitude.mean())
    amp_std = float(amplitude.std())
    phase_mean = float(phase.mean())
    phase_std = float(phase.std())

    if amp_std < 1e-12:
        amp_std = 1.0
    if phase_std < 1e-12:
        phase_std = 1.0

    amp_norm = (amplitude - amp_mean) / amp_std
    phase_norm = (phase - phase_mean) / phase_std

    stats = {
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
    }
    H_norm = np.stack([amp_norm, phase_norm], axis=-1)
    return H_norm.astype(np.float32), stats


def amplitude_phase_denormalize(H_norm: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    amp_norm = H_norm[..., 0]
    phase_norm = H_norm[..., 1]
    amplitude = amp_norm * stats["amp_std"] + stats["amp_mean"]
    phase = phase_norm * stats["phase_std"] + stats["phase_mean"]
    return amplitude * np.exp(1j * phase)


def build_amp_phase_features(H_freq: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    H_norm, amp_phase_stats = amplitude_phase_normalization(H_freq)
    N_tx, N_rx, N_sc, N_users, N_batches = H_freq.shape
    amp_phase_feature_tensor = np.zeros(
        (N_users * N_batches, 2 * N_tx * N_rx, N_sc), dtype=np.float32
    )

    for u_idx in range(N_users):
        for b_idx in range(N_batches):
            sample_idx = u_idx * N_batches + b_idx
            block = H_norm[:, :, :, u_idx, b_idx, :]  # [N_tx, N_rx, N_sc, 2]
            block = np.moveaxis(block, -1, 0)  # [2, N_tx, N_rx, N_sc]
            amp_phase_feature_tensor[sample_idx] = block.reshape(2 * N_tx * N_rx, N_sc)

    amp_phase_metadata = {
        "N_tx": N_tx,
        "N_rx": N_rx,
        "N_sc": N_sc,
        "N_users": N_users,
        "N_batches": N_batches,
        "feature_channels": amp_phase_feature_tensor.shape[1],
        "sequence_length": N_sc,
        "norm_stats": amp_phase_stats,
        "sample_order": "user-major (user*batches)",
    }
    return amp_phase_feature_tensor, amp_phase_metadata


def _normalize_pair(u_col: np.ndarray, v_col: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    u_col = np.asarray(u_col, dtype=np.complex64).copy()
    v_col = np.asarray(v_col, dtype=np.complex64).copy()
    u_norm = np.linalg.norm(u_col)
    if u_norm > eps:
        u_col /= u_norm
    v_norm = np.linalg.norm(v_col)
    if v_norm > eps:
        v_col /= v_norm
    pivot_indices = np.where(np.abs(u_col) > eps)[0]
    pivot_value = None
    if pivot_indices.size:
        pivot_value = u_col[pivot_indices[0]]
    else:
        pivot_indices = np.where(np.abs(v_col) > eps)[0]
        if pivot_indices.size:
            pivot_value = v_col[pivot_indices[0]]
    if pivot_value is not None:
        phase = np.exp(-1j * np.angle(pivot_value))
        u_col *= phase
        v_col *= phase
    return u_col, v_col


def _robust_svd(mat: np.ndarray, max_retries: int = _SVD_MAX_RETRIES, jitter_scale: float = 1e-6):
    """Compute an SVD with retries to avoid non-convergence."""

    mat_local = np.asarray(mat, dtype=np.complex128, order="C")
    for attempt in range(max_retries):
        try:
            return np.linalg.svd(mat_local, full_matrices=False)
        except np.linalg.LinAlgError:
            noise = jitter_scale * (attempt + 1)
            perturb = noise * (np.random.randn(*mat_local.shape) + 1j * np.random.randn(*mat_local.shape))
            mat_local = mat_local + perturb
    if torch is not None:
        mat_tensor = torch.from_numpy(mat_local.copy())
        U, S, Vh = torch.linalg.svd(mat_tensor.to(torch.complex128), full_matrices=False)
        return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()
    raise np.linalg.LinAlgError("SVD failed to converge even after retries.")


def _flatten_feature_block(arr: np.ndarray) -> np.ndarray:
    arr = arr.transpose(0, 2, 3, 1)
    batch, modes, features, seq = arr.shape
    return arr.reshape(batch, modes * features, seq)


def build_svd_feature_tensor(H_freq: np.ndarray, cfg: FeatureConfig) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    N_tx, N_rx, N_sc, N_users, N_batches = H_freq.shape
    min_dim = min(N_tx, N_rx)
    k_trunc = cfg.k_trunc or min_dim

    H_slices = H_freq.transpose(3, 4, 2, 0, 1)

    svd_U = np.zeros((N_users, N_batches, N_sc, N_tx, min_dim), dtype=np.complex64)
    svd_V = np.zeros((N_users, N_batches, N_sc, N_rx, min_dim), dtype=np.complex64)
    svd_S = np.zeros((N_users, N_batches, N_sc, min_dim), dtype=np.float32)
    svd_rank = np.zeros((N_users, N_batches, N_sc), dtype=np.int16)

    rank_values = []
    rank_counter: Dict[int, int] = {}
    degeneracy_counter: Dict[int, int] = {}

    for u_idx in range(N_users):
        for b_idx in range(N_batches):
            for sc_idx in range(N_sc):
                mat = H_slices[u_idx, b_idx, sc_idx]
                U, S, Vh = _robust_svd(mat)
                V = Vh.conj().T

                for col in range(min_dim):
                    u_col, v_col = _normalize_pair(U[:, col], V[:, col])
                    U[:, col] = u_col
                    V[:, col] = v_col

                svd_U[u_idx, b_idx, sc_idx] = U[:, :min_dim].astype(np.complex64)
                svd_V[u_idx, b_idx, sc_idx] = V[:, :min_dim].astype(np.complex64)
                svd_S[u_idx, b_idx, sc_idx] = S.astype(np.float32)

                denom = S[0] if S.size else 0.0
                rel = S / (denom + 1e-12) if denom > 0 else np.zeros_like(S)
                effective_rank = int((rel > cfg.tau_rank_rel).sum())
                if effective_rank == 0 and S.size:
                    effective_rank = 1
                rank_values.append(effective_rank)
                rank_counter[effective_rank] = rank_counter.get(effective_rank, 0) + 1
                svd_rank[u_idx, b_idx, sc_idx] = effective_rank

                groups = []
                if S.size:
                    start = 0
                    for idx in range(S.size - 1):
                        gap = abs(S[idx] - S[idx + 1])
                        denom_gap = max(abs(S[idx]), 1e-12)
                        if gap / denom_gap > cfg.tau_degen_rel:
                            groups.append(idx - start + 1)
                            start = idx + 1
                    groups.append(S.size - start)
                else:
                    groups.append(0)
                for g in groups:
                    degeneracy_counter[g] = degeneracy_counter.get(g, 0) + 1

    rank_array = np.asarray(rank_values, dtype=np.int16)

    num_samples = N_users * N_batches
    sequence_length = N_sc

    channels_per_mode = 4 * (N_tx + N_rx) + 1
    amp_u = np.zeros((num_samples, sequence_length, k_trunc, N_tx), dtype=np.float32)
    phase_u = np.zeros_like(amp_u)
    amp_v = np.zeros((num_samples, sequence_length, k_trunc, N_rx), dtype=np.float32)
    phase_v = np.zeros_like(amp_v)
    sigma_raw = np.zeros((num_samples, sequence_length, k_trunc), dtype=np.float32)
    sigma_valid = np.zeros_like(sigma_raw, dtype=bool)
    sample_short_rank = np.zeros(num_samples, dtype=bool)

    for u_idx in range(N_users):
        for b_idx in range(N_batches):
            sample_idx = u_idx * N_batches + b_idx
            for sc_idx in range(N_sc):
                singular_values = svd_S[u_idx, b_idx, sc_idx]
                U_block = svd_U[u_idx, b_idx, sc_idx]
                V_block = svd_V[u_idx, b_idx, sc_idx]
                take = min(k_trunc, singular_values.size)
                if take < k_trunc:
                    sample_short_rank[sample_idx] = True
                if take == 0:
                    continue
                amp_u[sample_idx, sc_idx, :take, :] = np.abs(U_block[:, :take].T)
                phase_u[sample_idx, sc_idx, :take, :] = np.angle(U_block[:, :take].T)
                amp_v[sample_idx, sc_idx, :take, :] = np.abs(V_block[:, :take].T)
                phase_v[sample_idx, sc_idx, :take, :] = np.angle(V_block[:, :take].T)
                sigma_raw[sample_idx, sc_idx, :take] = singular_values[:take]
                sigma_valid[sample_idx, sc_idx, :take] = True

    if not cfg.allow_zero_padding and sample_short_rank.any():
        missing = int(sample_short_rank.sum())
        raise ValueError(
            f"k_trunc={k_trunc} exceeds the minimal rank for {missing} samples; "
            f"set allow_zero_padding=True or reduce k_trunc."
        )

    sigma_values = sigma_raw[sigma_valid]
    if sigma_values.size:
        sigma_mean = float(sigma_values.mean())
        sigma_std = float(sigma_values.std())
    else:
        sigma_mean = 0.0
        sigma_std = 1.0
    if sigma_std < 1e-12:
        sigma_std = 1.0

    sigma_z = np.zeros_like(sigma_raw, dtype=np.float32)
    if sigma_values.size:
        sigma_z[sigma_valid] = (sigma_raw[sigma_valid] - sigma_mean) / sigma_std
    sigma_padding_token = float((0.0 - sigma_mean) / sigma_std)
    sigma_z[~sigma_valid] = sigma_padding_token

    amp_u_block = _flatten_feature_block(amp_u)
    phase_u_block = _flatten_feature_block(phase_u)
    amp_v_block = _flatten_feature_block(amp_v)
    phase_v_block = _flatten_feature_block(phase_v)
    sigma_block = _flatten_feature_block(sigma_z[..., None])

    svd_feature_tensor = np.concatenate(
        [amp_u_block, phase_u_block, amp_v_block, phase_v_block, sigma_block],
        axis=1,
    ).astype(np.float32)

    feature_channels = svd_feature_tensor.shape[1]
    padding_fraction = sample_short_rank.sum() / num_samples if num_samples else 0.0

    svd_summary = {
        "tau_rank_rel": cfg.tau_rank_rel,
        "tau_degen_rel": cfg.tau_degen_rel,
        "rank_hist": dict(rank_counter),
        "degeneracy_hist": dict(degeneracy_counter),
        "rank_array": rank_array,
        "padding_fraction": padding_fraction,
    }

    svd_feature_metadata = {
        "N_tx": N_tx,
        "N_rx": N_rx,
        "N_sc": N_sc,
        "N_users": N_users,
        "N_batches": N_batches,
        "feature_channels": feature_channels,
        "sequence_length": sequence_length,
        "k_trunc": k_trunc,
        "allow_zero_padding": cfg.allow_zero_padding,
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "sigma_padding_token": sigma_padding_token,
        "channels_per_mode": channels_per_mode,
    }

    return svd_feature_tensor, svd_feature_metadata, svd_summary


def extract_feature_tensors(H_freq: np.ndarray, cfg: FeatureConfig) -> FeatureExtractionResult:
    amp_phase_features, amp_phase_metadata = build_amp_phase_features(H_freq)
    svd_features, svd_metadata, svd_summary = build_svd_feature_tensor(H_freq, cfg)
    return FeatureExtractionResult(
        amp_phase_features=amp_phase_features,
        amp_phase_metadata=amp_phase_metadata,
        svd_features=svd_features,
        svd_metadata=svd_metadata,
        svd_summary=svd_summary,
    )


def flatten_feature_tensor_for_swd(
    feature_tensor: ArrayLike, max_points: int = 200_000, seed: int = 0
) -> np.ndarray:
    if isinstance(feature_tensor, torch.Tensor):
        tensor = feature_tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(feature_tensor)
    samples = tensor.shape[0]
    rng = np.random.default_rng(seed)
    if tensor.ndim == 2:
        flattened = tensor
    elif tensor.ndim >= 3:
        flattened = tensor.reshape(samples, -1)
    else:
        raise ValueError("feature_tensor must have ndim >= 2")
    if flattened.shape[0] > max_points:
        idx = rng.choice(flattened.shape[0], size=max_points, replace=False)
        flattened = flattened[idx]
    return flattened.astype(np.float32)


def unpack_feature_tensor(
    feature_tensor: ArrayLike, metadata: Dict[str, Any], denormalize_sigma: bool = False
) -> Dict[str, np.ndarray]:
    if isinstance(feature_tensor, torch.Tensor):
        tensor = feature_tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(feature_tensor)

    channels = metadata["feature_channels"]
    seq = metadata["sequence_length"]
    k_trunc = metadata["k_trunc"]
    N_tx = metadata["N_tx"]
    N_rx = metadata["N_rx"]

    amp_u = tensor[:, : N_tx * k_trunc, :].reshape(-1, seq, k_trunc, N_tx)
    phase_u = tensor[:, N_tx * k_trunc : 2 * N_tx * k_trunc, :].reshape(-1, seq, k_trunc, N_tx)

    start = 2 * N_tx * k_trunc
    stop = start + N_rx * k_trunc
    amp_v = tensor[:, start:stop, :].reshape(-1, seq, k_trunc, N_rx)
    phase_v = tensor[:, stop : stop + N_rx * k_trunc, :].reshape(-1, seq, k_trunc, N_rx)
    sigma_start = 2 * (N_tx + N_rx) * k_trunc
    sigma = tensor[:, sigma_start:, :].reshape(-1, seq, k_trunc)

    if denormalize_sigma:
        sigma = sigma * metadata["sigma_std"] + metadata["sigma_mean"]

    return {
        "amp_u": amp_u,
        "phase_u": phase_u,
        "amp_v": amp_v,
        "phase_v": phase_v,
        "sigma": sigma,
    }


def features_to_svd_components(
    feature_tensor: ArrayLike,
    metadata: Dict[str, Any],
    denormalize_sigma: bool = True,
    enforce_unit_norm: bool = True,
) -> Dict[str, np.ndarray]:
    components = unpack_feature_tensor(feature_tensor, metadata, denormalize_sigma=denormalize_sigma)
    for key in ("amp_u", "amp_v", "phase_u", "phase_v"):
        components[key] = np.asarray(components[key], dtype=np.float32)

    if enforce_unit_norm:
        def _normalize(arr_amp, arr_phase):
            complex_arr = arr_amp * np.exp(1j * arr_phase)
            norms = np.maximum(np.linalg.norm(complex_arr, axis=-1, keepdims=True), 1e-8)
            return complex_arr / norms

        components["U"] = _normalize(components["amp_u"], components["phase_u"])
        components["V"] = _normalize(components["amp_v"], components["phase_v"])
    else:
        components["U"] = components["amp_u"] * np.exp(1j * components["phase_u"])
        components["V"] = components["amp_v"] * np.exp(1j * components["phase_v"])

    components["sigma"] = np.clip(components["sigma"], 0.0, None).astype(np.float32)
    return components


def reconstruct_channel_from_features(
    feature_tensor: ArrayLike, metadata: Dict[str, Any]
) -> np.ndarray:
    components = features_to_svd_components(
        feature_tensor, metadata, denormalize_sigma=True, enforce_unit_norm=True
    )
    U = components["U"]  # [B, seq, K, N_tx]
    V = components["V"]  # [B, seq, K, N_rx]
    sigma = components["sigma"]  # [B, seq, K]
    recon = np.einsum("bskn,bskm,bsk->bnms", U, np.conjugate(V), sigma, optimize=True)
    return recon.astype(np.complex64)


def amp_phase_features_to_complex(feature_tensor: ArrayLike, metadata: Dict[str, Any]) -> np.ndarray:
    if isinstance(feature_tensor, torch.Tensor):
        tensor = feature_tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(feature_tensor)
    if tensor.ndim != 3:
        raise ValueError(f"Expected feature tensor with 3 dims, got {tensor.shape}")
    batch, channels, seq = tensor.shape
    expected_channels = 2 * metadata["N_tx"] * metadata["N_rx"]
    if channels != expected_channels or seq != metadata["N_sc"]:
        raise ValueError(
            f"Unexpected feature tensor shape {tensor.shape}, expected "
            f"({batch}, {expected_channels}, {metadata['N_sc']})"
        )
    arr = tensor.reshape(batch, 2, metadata["N_tx"], metadata["N_rx"], seq)
    arr = np.moveaxis(arr, 1, -1)  # [B, N_tx, N_rx, seq, 2]
    amp = arr[..., 0]
    phase = arr[..., 1]
    stats = metadata["norm_stats"]
    amplitude = amp * stats["amp_std"] + stats["amp_mean"]
    phase_val = phase * stats["phase_std"] + stats["phase_mean"]
    return amplitude * np.exp(1j * phase_val).astype(np.complex64)


def complex_to_svd_feature_tensor(H_complex: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    if isinstance(H_complex, torch.Tensor):
        H_complex = H_complex.detach().cpu().numpy()
    else:
        H_complex = np.asarray(H_complex)
    if H_complex.ndim != 4:
        raise ValueError(f"Expected complex tensor with 4 dims [B, N_tx, N_rx, N_sc], got {H_complex.shape}")
    num_samples, N_tx_local, N_rx_local, N_sc_local = H_complex.shape
    if (N_tx_local, N_rx_local, N_sc_local) != (
        metadata["N_tx"],
        metadata["N_rx"],
        metadata["N_sc"],
    ):
        raise ValueError("Input dimensions do not match metadata")

    K = metadata["k_trunc"]
    sigma_mean = metadata["sigma_mean"]
    sigma_std = metadata["sigma_std"] if metadata["sigma_std"] >= 1e-12 else 1.0
    sigma_padding_token = metadata["sigma_padding_token"]

    amp_u_tmp = np.zeros((num_samples, N_sc_local, K, N_tx_local), dtype=np.float32)
    phase_u_tmp = np.zeros_like(amp_u_tmp)
    amp_v_tmp = np.zeros((num_samples, N_sc_local, K, N_rx_local), dtype=np.float32)
    phase_v_tmp = np.zeros_like(amp_v_tmp)
    sigma_z_tmp = np.full((num_samples, N_sc_local, K), sigma_padding_token, dtype=np.float32)

    for sample_idx in range(num_samples):
        for sc_idx in range(N_sc_local):
            mat = H_complex[sample_idx, :, :, sc_idx]
            U, S, Vh = _robust_svd(mat)
            V = Vh.conj().T
            take = min(K, S.size)
            for col in range(take):
                u_col, v_col = _normalize_pair(U[:, col], V[:, col])
                U[:, col] = u_col
                V[:, col] = v_col
            if take:
                amp_u_tmp[sample_idx, sc_idx, :take] = np.abs(U[:, :take].T)
                phase_u_tmp[sample_idx, sc_idx, :take] = np.angle(U[:, :take].T)
                amp_v_tmp[sample_idx, sc_idx, :take] = np.abs(V[:, :take].T)
                phase_v_tmp[sample_idx, sc_idx, :take] = np.angle(V[:, :take].T)
                sigma_z_tmp[sample_idx, sc_idx, :take] = (S[:take] - sigma_mean) / sigma_std

    amp_block = _flatten_feature_block(amp_u_tmp)
    phase_u_block = _flatten_feature_block(phase_u_tmp)
    amp_v_block = _flatten_feature_block(amp_v_tmp)
    phase_v_block = _flatten_feature_block(phase_v_tmp)
    sigma_block = _flatten_feature_block(sigma_z_tmp[..., None])
    features = np.concatenate(
        [amp_block, phase_u_block, amp_v_block, phase_v_block, sigma_block],
        axis=1,
    )
    return features.astype(np.float32)
