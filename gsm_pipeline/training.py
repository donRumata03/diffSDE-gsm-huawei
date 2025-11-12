from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .bridges.repo_solvers import (
    DSB,
    DSBM,
    RectifiedFlow,
    ScoreNetwork,
    train_dsb_ipf as repo_train_dsb_ipf,
    train_dsbm as repo_train_dsbm,
    train_flow_model as repo_train_flow_model,
)
from .config import TrainingConfig


@dataclass
class TrainingResult:
    model: nn.Module
    optimizer_state: Dict[str, torch.Tensor]
    history: List[float]
    method: str
    device: torch.device


@dataclass
class RepoBridgeResult:
    solver_name: str
    forward_model: nn.Module
    history: List[float]
    device: torch.device
    feature_shape: Tuple[int, int]
    flatten_dim: int

    def sample(self, num_samples: int, steps: Optional[int] = None) -> torch.Tensor:
        shape = (num_samples, self.flatten_dim)
        latent = torch.randn(shape, device=self.device)
        if self.solver_name in {"dsb", "dsbm"}:
            traj = self.forward_model.sample_sde(latent, fb="f", steps=steps)
        else:
            traj = self.forward_model.sample_ode(latent, fb="f", steps=steps)
        samples = traj[-1].cpu().view(num_samples, *self.feature_shape)
        return samples

    def score(self, x: torch.Tensor, t_value: float = 0.5) -> torch.Tensor:
        flat = x.view(x.size(0), -1).to(self.device)
        t = torch.full((flat.size(0), 1), t_value, device=self.device)
        if self.solver_name in {"dsb", "dsbm"}:
            preds = self.forward_model.net_dict["f"](flat, t)
        else:
            preds = self.forward_model.net(flat, t)
        return preds.detach().cpu().view(x.size(0), *self.feature_shape)


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_feature_dataloaders(
    feature_tensor: np.ndarray,
    batch_size: int,
    val_split: float,
    seed: int = 0,
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], torch.Tensor, Optional[torch.Tensor]]:
    tensor = torch.from_numpy(np.asarray(feature_tensor, dtype=np.float32))
    num_samples = tensor.shape[0]
    if num_samples == 0:
        raise ValueError("Feature tensor is empty.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(round(num_samples * val_split))
    val_size = min(max(val_size, 0), num_samples)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:] if val_size < num_samples else indices[val_size:]

    if train_indices.size == 0:
        raise ValueError("Training split is empty; reduce val_split.")

    train_tensor = tensor[train_indices]
    val_tensor = tensor[val_indices] if val_size else None

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = None
    if val_tensor is not None and len(val_tensor) > 0:
        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return train_loader, val_loader, train_tensor, val_tensor


def _progress(iterable: Iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def _cfm_call_and_reshape(cfm_obj, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x1.shape[0]
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)

    try:
        out = cfm_obj.sample_location_and_conditional_flow(x0_flat, x1_flat, t)
    except AttributeError:
        out = cfm_obj(x0_flat, x1_flat, t)

    if isinstance(out, dict):
        xt_flat = out.get("xt")
        ut_flat = out.get("ut")
        if xt_flat is None or ut_flat is None:
            tensors = [v for v in out.values() if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == B]
            xt_flat, ut_flat = tensors[0], tensors[1]
    else:
        seq = list(out) if isinstance(out, (tuple, list)) else [out]
        tensors = [v for v in seq if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == B]
        xt_flat, ut_flat = tensors[0], tensors[1]

    xt = xt_flat.view_as(x1)
    ut = ut_flat.view_as(x1)
    return xt, ut


def train_sb_cfm(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    try:
        from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
    except Exception:
        from torchcfm.losses import SchrodingerBridgeConditionalFlowMatcher

    cfm = SchrodingerBridgeConditionalFlowMatcher(
        sigma=config.cfm_sigma,
        ot_method=config.cfm_ot_method,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: List[float] = []
    for epoch in range(config.num_epochs):
        total = 0.0
        iterator = _progress(train_loader, config.tqdm, f"SB-CFM epoch {epoch + 1}/{config.num_epochs}")
        for (x1_batch,) in iterator:
            x1 = x1_batch.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=device)

            xt, ut = _cfm_call_and_reshape(cfm, x0, x1, t)
            xt, ut = xt.to(device), ut.to(device)

            pred = model(xt, t)
            loss = F.mse_loss(pred, ut)

            opt.zero_grad()
            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
            opt.step()
            total += loss.item()

        epoch_loss = total / len(train_loader)
        history.append(epoch_loss)
        print(f"[SB-CFM] Epoch {epoch + 1}: loss={epoch_loss:.6f}")

    return TrainingResult(
        model=model,
        optimizer_state=opt.state_dict(),
        history=history,
        method="sb-cfm",
        device=device,
    )


def train_dsbm(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    sigma_min = max(config.dsbm_sigma_min, 1e-4)
    sigma_max = max(config.dsbm_sigma_max, sigma_min + 1e-4)
    log_sigma_min = np.log(sigma_min)
    log_sigma_max = np.log(sigma_max)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: List[float] = []
    for epoch in range(config.num_epochs):
        total = 0.0
        iterator = _progress(train_loader, config.tqdm, f"DSBM epoch {epoch + 1}/{config.num_epochs}")
        for (batch,) in iterator:
            x = batch.to(device)
            u = torch.rand(x.size(0), device=device)
            log_sigma = log_sigma_min + u * (log_sigma_max - log_sigma_min)
            sigma = torch.exp(log_sigma)
            noise = torch.randn_like(x) * sigma.view(-1, 1, 1)
            perturbed = x + noise
            t = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)

            pred = model(perturbed, t)
            target = -noise / (sigma.view(-1, 1, 1) ** 2)
            loss = 0.5 * (pred - target).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
            opt.step()
            total += loss.item()

        epoch_loss = total / len(train_loader)
        history.append(epoch_loss)
        print(f"[DSBM] Epoch {epoch + 1}: loss={epoch_loss:.6f}")

    return TrainingResult(
        model=model,
        optimizer_state=opt.state_dict(),
        history=history,
        method="dsbm",
        device=device,
    )


def train_repo_bridge(
    train_tensor: torch.Tensor,
    config: TrainingConfig,
) -> RepoBridgeResult:
    if train_tensor.dim() != 3:
        raise ValueError(f"train_tensor shape {train_tensor.shape} is not [N, C, L].")
    device = resolve_device(config.device)
    batch, channels, length = train_tensor.shape
    feature_shape = (channels, length)
    flatten_dim = channels * length
    flattened = train_tensor.view(batch, flatten_dim).detach().cpu()
    gaussian = torch.randn_like(flattened)
    x_pairs = torch.stack([gaussian, flattened], dim=1)

    input_dim = flatten_dim + 1
    hidden_dims = list(config.bridge_hidden_dims)
    if not hidden_dims or hidden_dims[-1] != flatten_dim:
        layer_widths = hidden_dims + [flatten_dim]
    else:
        layer_widths = hidden_dims

    def build_net() -> nn.Module:
        net = ScoreNetwork(
            input_dim=input_dim,
            layer_widths=layer_widths,
            activation=config.bridge_activation,
        )
        return net.to(device)

    solver_key = config.bridge_solver.lower()
    if solver_key == "dsb":
        solver = DSB(
            net_fwd=build_net(),
            net_bwd=build_net(),
            num_steps=config.bridge_num_steps,
            sigma=config.bridge_sigma,
            device=device,
        )
    elif solver_key == "dsbm":
        solver = DSBM(
            net_fwd=build_net(),
            net_bwd=build_net(),
            num_steps=config.bridge_num_steps,
            sigma=config.bridge_sigma,
            eps=config.bridge_eps,
            first_coupling=config.bridge_first_coupling,
            device=device,
        )
    elif solver_key in {"rectifiedflow", "fm", "flowmatching"}:
        solver = RectifiedFlow(
            net=build_net(),
            num_steps=config.bridge_num_steps,
            sigma=config.bridge_sigma,
            device=device,
        )
        solver_key = "rectifiedflow"
    else:
        raise ValueError(f"Unsupported bridge_solver '{config.bridge_solver}'.")

    history: List[float] = []
    trained_models: List[Dict[str, nn.Module]] = []
    total_passes = config.bridge_outer_iters * len(config.bridge_fb_sequence)
    pass_counter = 0

    for _ in range(config.bridge_outer_iters):
        for fb in config.bridge_fb_sequence:
            pass_counter += 1
            first_pass = pass_counter == 1
            prev_entry = trained_models[-1] if trained_models else None
            prev_model = prev_entry["model"] if prev_entry else None
            solver.fb = fb
            if solver_key == "dsb":
                solver, losses = repo_train_dsb_ipf(
                    solver,
                    x_pairs,
                    batch_size=config.batch_size,
                    inner_iters=config.bridge_inner_iters,
                    learning_rate=config.learning_rate,
                    fb=fb,
                    first_it=first_pass,
                )
            elif solver_key == "dsbm":
                solver, losses = repo_train_dsbm(
                    solver,
                    x_pairs,
                    batch_size=config.batch_size,
                    inner_iters=config.bridge_inner_iters,
                    learning_rate=config.learning_rate,
                    fb=fb,
                    first_it=first_pass,
                    prev_model=prev_model,
                )
            else:
                solver, losses = repo_train_flow_model(
                    solver,
                    x_pairs,
                    batch_size=config.batch_size,
                    inner_iters=config.bridge_inner_iters,
                    learning_rate=config.learning_rate,
                    fb=fb,
                    first_it=first_pass,
                    prev_model=prev_model,
                )
            history.extend(losses)
            stored = deepcopy(solver)
            stored.fb = fb
            trained_models.append({"fb": fb, "model": stored})
            if pass_counter >= total_passes:
                break
        if pass_counter >= total_passes:
            break

    if not trained_models:
        raise RuntimeError("Bridge training produced no models.")
    forward_entry = next((entry for entry in reversed(trained_models) if entry["fb"] == "f"), trained_models[-1])
    forward_model = forward_entry["model"].to(device)

    return RepoBridgeResult(
        solver_name=solver_key,
        forward_model=forward_model,
        history=history,
        device=device,
        feature_shape=feature_shape,
        flatten_dim=flatten_dim,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    device = device or resolve_device(config.device)
    model = model.to(device)
    if config.method.lower() == "sb-cfm":
        return train_sb_cfm(model, train_loader, config, device)
    if config.method.lower() == "dsbm":
        return train_dsbm(model, train_loader, config, device)
    raise ValueError("Unsupported method. Use 'dsbm' or 'sb-cfm'.")
