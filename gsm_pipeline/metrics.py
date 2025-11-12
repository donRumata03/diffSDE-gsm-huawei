from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ot.sliced import sliced_wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

from .config import SamplingConfig, ScoreEvalConfig
from .features import (
    features_to_svd_components,
    flatten_feature_tensor_for_swd,
    unpack_feature_tensor,
)


def compute_swd_between_features(
    real_features,
    gen_features,
    *,
    n_projections: int = 128,
    max_points: int = 200_000,
    seed_real: int = 0,
    seed_gen: int = 1,
) -> float:
    X = flatten_feature_tensor_for_swd(real_features, max_points=max_points, seed=seed_real)
    Y = flatten_feature_tensor_for_swd(gen_features, max_points=max_points, seed=seed_gen)
    return float(sliced_wasserstein_distance(X, Y, n_projections=n_projections, p=1, seed=42))


def summarize_component_stats(label: str, array) -> Dict[str, float]:
    vals = np.asarray(array).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return {"label": label, "mean": float("nan")}
    return {
        "label": label,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def evaluate_svd_statistics(real_features, gen_features, metadata) -> Dict[str, Dict[str, float]]:
    real = unpack_feature_tensor(real_features, metadata, denormalize_sigma=True)
    gen = unpack_feature_tensor(gen_features, metadata, denormalize_sigma=True)
    stats = {}
    stats["sigma_real"] = summarize_component_stats("sigma_real", real["sigma"][real["sigma"] > 0])
    stats["sigma_gen"] = summarize_component_stats("sigma_gen", gen["sigma"][gen["sigma"] > 0])
    stats["amp_u_real"] = summarize_component_stats("amp_u_real", real["amp_u"])
    stats["amp_u_gen"] = summarize_component_stats("amp_u_gen", gen["amp_u"])
    stats["phase_u_real"] = summarize_component_stats("phase_u_real", real["phase_u"])
    stats["phase_u_gen"] = summarize_component_stats("phase_u_gen", gen["phase_u"])
    stats["amp_v_real"] = summarize_component_stats("amp_v_real", real["amp_v"])
    stats["amp_v_gen"] = summarize_component_stats("amp_v_gen", gen["amp_v"])
    stats["phase_v_real"] = summarize_component_stats("phase_v_real", real["phase_v"])
    stats["phase_v_gen"] = summarize_component_stats("phase_v_gen", gen["phase_v"])
    return stats


def _collect_grassmann_metrics(U_real, U_gen, sigma_real, sigma_gen, sigma_threshold=1e-6, transpose_vectors=False):
    min_eigs = []
    spectra = []
    batch, seq, k, dim = U_real.shape
    for b in range(batch):
        for s in range(seq):
            sigma_r = sigma_real[b, s]
            sigma_g = sigma_gen[b, s]
            valid_r = sigma_r > sigma_threshold
            valid_g = sigma_g > sigma_threshold
            k_valid = min(np.count_nonzero(valid_r), np.count_nonzero(valid_g))
            if k_valid == 0:
                continue
            idx_r = np.nonzero(valid_r)[0][:k_valid]
            idx_g = np.nonzero(valid_g)[0][:k_valid]
            if transpose_vectors:
                mat_real = U_real[b, s].astype(np.complex128).transpose(1, 0)
                mat_gen = U_gen[b, s].astype(np.complex128).transpose(1, 0)
                X_real = mat_real[:, idx_r]
                X_gen = mat_gen[:, idx_g]
            else:
                X_real = U_real[b, s, idx_r].astype(np.complex128).T
                X_gen = U_gen[b, s, idx_g].astype(np.complex128).T
            if X_real.size == 0 or X_gen.size == 0:
                continue
            Q_real, _ = np.linalg.qr(X_real, mode="reduced")
            Q_gen, _ = np.linalg.qr(X_gen, mode="reduced")
            k_eff = min(Q_real.shape[1], Q_gen.shape[1])
            if k_eff == 0:
                continue
            Q_real = Q_real[:, :k_eff]
            Q_gen = Q_gen[:, :k_eff]
            overlap = Q_real.conj().T @ Q_gen
            gram = overlap.conj().T @ overlap
            eigvals = np.linalg.eigvalsh(gram)
            eigvals = np.clip(eigvals.real, 0.0, None)
            if eigvals.size == 0:
                continue
            min_eigs.append(float(eigvals.min()))
            spectra.append(eigvals)
    return np.asarray(min_eigs, dtype=np.float64), spectra


def evaluate_grassmann_metric(real_features, gen_features, metadata) -> Dict[str, float]:
    svd_real = features_to_svd_components(
        real_features, metadata, denormalize_sigma=True, enforce_unit_norm=True
    )
    svd_gen = features_to_svd_components(
        gen_features, metadata, denormalize_sigma=True, enforce_unit_norm=True
    )
    min_eigs_u, spectra_u = _collect_grassmann_metrics(
        svd_real["U"], svd_gen["U"], svd_real["sigma"], svd_gen["sigma"], transpose_vectors=True
    )
    min_eigs_v, spectra_v = _collect_grassmann_metrics(
        svd_real["V"], svd_gen["V"], svd_real["sigma"], svd_gen["sigma"]
    )
    metrics = {}
    metrics["mean_min_eig_U"] = float(min_eigs_u.mean()) if min_eigs_u.size else float("nan")
    metrics["mean_min_eig_V"] = float(min_eigs_v.mean()) if min_eigs_v.size else float("nan")
    metrics["grassmann_metric"] = np.nanmean([metrics["mean_min_eig_U"], metrics["mean_min_eig_V"]])
    metrics["spectra_U"] = spectra_u
    metrics["spectra_V"] = spectra_v
    return metrics


def collect_score_distribution(
    model: Optional[torch.nn.Module],
    data_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
    t_value: float = 0.5,
    score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    results: List[torch.Tensor] = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            x = batch.to(device)
            t = torch.full((x.size(0),), t_value, device=device)
            if score_fn is not None:
                score = score_fn(x, t)
            elif model is not None:
                score = model(x, t)
            else:
                raise ValueError("Either model or score_fn must be provided.")
            results.append(score.detach().cpu())
    return torch.cat(results, dim=0)


def sliced_wasserstein_between_scores(
    real_scores: torch.Tensor,
    gen_scores: torch.Tensor,
    cfg: ScoreEvalConfig,
) -> float:
    X = flatten_feature_tensor_for_swd(real_scores, max_points=cfg.max_points, seed=cfg.seed_real)
    Y = flatten_feature_tensor_for_swd(gen_scores, max_points=cfg.max_points, seed=cfg.seed_gen)
    return float(
        sliced_wasserstein_distance(
            X,
            Y,
            n_projections=cfg.n_projections,
            p=1,
            seed=42,
        )
    )


def score_distribution_summary(scores: torch.Tensor) -> Dict[str, float]:
    arr = scores.detach().cpu().numpy().reshape(-1)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"mean": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def sweep_nfe_and_metrics(
    sampler_fn: Callable[..., torch.Tensor],
    model: torch.nn.Module,
    real_features: torch.Tensor,
    metadata: Dict[str, any],
    *,
    steps_list: Sequence[int],
    sampling_cfg: SamplingConfig,
    train_space: str,
    amp_phase_metadata: Optional[Dict[str, any]] = None,
    n_projections: int = 128,
    max_points: int = 200_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_flat = flatten_feature_tensor_for_swd(real_features, max_points=max_points, seed=0)
    real_sigma = unpack_feature_tensor(real_features, metadata, denormalize_sigma=True)["sigma"]
    real_sigma_vals = real_sigma[real_sigma > 0]
    target_sigma_std = float(real_sigma_vals.std()) if real_sigma_vals.size else 0.0

    sigma_stds, swds = [], []
    device = sampling_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for steps in steps_list:
        gen_features = sampler_fn(
            model,
            num_samples=sampling_cfg.num_samples,
            shape=(real_features.shape[1], real_features.shape[2]),
            steps=steps,
            sde_sigma=sampling_cfg.sde_sigma,
            schedule=sampling_cfg.schedule,
            device=device,
        )

        if train_space == "svd":
            gen_svd_chunk = gen_features.detach().cpu()
        else:
            from .features import amp_phase_features_to_complex, complex_to_svd_feature_tensor

            gen_complex = amp_phase_features_to_complex(gen_features, amp_phase_metadata)
            gen_svd_np = complex_to_svd_feature_tensor(gen_complex, metadata)
            gen_svd_chunk = torch.from_numpy(gen_svd_np)

        sigma_chunk = unpack_feature_tensor(gen_svd_chunk, metadata, denormalize_sigma=True)["sigma"]
        sigma_vals = sigma_chunk[sigma_chunk > 0].reshape(-1)
        sigma_std = float(sigma_vals.std()) if sigma_vals.size else 0.0

        chunk_flat = flatten_feature_tensor_for_swd(gen_svd_chunk, max_points=max_points, seed=steps)
        swd_chunk = float(sliced_wasserstein_distance(real_flat, chunk_flat, n_projections=n_projections, p=1, seed=42))

        sigma_stds.append(sigma_std)
        swds.append(swd_chunk)
        if verbose:
            print(
                f"NFE={steps:4d} | sigma std={sigma_std:.4f} (target {target_sigma_std:.4f}) | "
                f"SWD={swd_chunk:.6f}"
            )

    return np.asarray(steps_list), np.asarray(sigma_stds), np.asarray(swds)
