from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def _resolve_activation(name_or_module: Optional[str | nn.Module]) -> nn.Module:
    """Return an activation module for the provided identifier."""

    if isinstance(name_or_module, nn.Module):
        return name_or_module
    if name_or_module is None:
        return nn.SiLU()
    name = str(name_or_module).lower()
    mapping = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'.")
    return mapping[name]


class ScoreNetwork(nn.Module):
    """Time-conditioned MLP used inside the DSBM repository solvers.

    Adapted from `dsbm-pytorch/DSBM-Gaussian.py`.
    """

    def __init__(
        self,
        input_dim: int,
        layer_widths: Sequence[int],
        activation: Optional[str | nn.Module] = "silu",
    ) -> None:
        super().__init__()
        if not layer_widths:
            raise ValueError("layer_widths must be non-empty.")
        layers = []
        prev_width = input_dim
        for hidden in layer_widths[:-1]:
            layers.append(nn.Linear(prev_width, hidden))
            layers.append(_resolve_activation(activation))
            prev_width = hidden
        layers.append(nn.Linear(prev_width, layer_widths[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


class DSB(nn.Module):
    """Original Diffusion Schrödinger Bridge (IPF) solver."""

    def __init__(
        self,
        net_fwd: nn.Module,
        net_bwd: nn.Module,
        num_steps: int,
        sigma: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd
        self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
        self.N = num_steps
        self.sigma = sigma
        self.device = device
        self.fb: Optional[str] = None

    @torch.no_grad()
    def generate_new_dataset_and_train_tuple(
        self,
        x_pairs: torch.Tensor,
        fb: str,
        first_it: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert fb in {"f", "b"}
        prev_fb = "b" if fb == "f" else "f"
        zstart = x_pairs[:, 1].to(self.device) if fb == "f" else x_pairs[:, 0].to(self.device)
        N = self.N
        dt = 1.0 / N
        z = zstart.detach().clone()
        batchsize = z.shape[0]
        dim = z.shape[1]
        ts = np.arange(N, dtype=np.float32) / N
        tl = np.arange(1, N + 1, dtype=np.float32) / N
        if prev_fb == "b":
            ts = 1 - ts
            tl = 1 - tl
        rand_idx = torch.randint(N, (batchsize,), device=self.device)
        z_selected = torch.zeros_like(z)
        t_selected = torch.zeros(batchsize, 1, device=self.device)
        target_selected = torch.zeros_like(z)

        if first_it:
            assert prev_fb == "f"
            for i in range(N):
                dz = self.sigma * torch.randn_like(z) * np.sqrt(dt)
                z = z + dz
                signal_val = -dz
                mask = rand_idx == i
                if mask.any():
                    z_selected[mask] = z.detach().clone()[mask]
                    t_selected[mask] = torch.full((mask.sum(), 1), float(tl[i]), device=self.device)
                    target_selected[mask] = signal_val[mask]
        else:
            for i in range(N):
                t = torch.full((batchsize, 1), ts[i], device=self.device)
                pred = self.net_dict[prev_fb](z, t)
                z = z.detach().clone() + pred
                dz = self.sigma * torch.randn_like(z) * np.sqrt(dt)
                z = z + dz
                signal_val = -self.net_dict[prev_fb](z, t) - dz
                mask = rand_idx == i
                if mask.any():
                    z_selected[mask] = z.detach().clone()[mask]
                    t_selected[mask] = torch.full((mask.sum(), 1), float(tl[i]), device=self.device)
                    target_selected[mask] = signal_val[mask]

        return z_selected.cpu(), t_selected.cpu(), target_selected.cpu()

    @torch.no_grad()
    def sample_sde(
        self,
        zstart: torch.Tensor,
        fb: str = "f",
        steps: Optional[int] = None,
    ) -> list[torch.Tensor]:
        assert fb in {"f", "b"}
        N = steps or self.N
        dt = 1.0 / N
        traj = []
        z = zstart.to(self.device).detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        ts = np.arange(N, dtype=np.float32) / N
        if fb == "b":
            ts = 1 - ts
        for i in range(N):
            t = torch.full((batchsize, 1), ts[i], device=self.device)
            pred = self.net_dict[fb](z, t)
            z = z.detach().clone() + pred
            z = z + self.sigma * torch.randn_like(z) * np.sqrt(dt)
            traj.append(z.detach().clone())
        return [state.cpu() for state in traj]


class DSBM(nn.Module):
    """Diffusion Schrödinger Bridge Matching solver."""

    def __init__(
        self,
        net_fwd: nn.Module,
        net_bwd: nn.Module,
        num_steps: int,
        sigma: float,
        eps: float,
        first_coupling: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd
        self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
        self.N = num_steps
        self.sigma = sigma
        self.eps = eps
        self.first_coupling = first_coupling
        self.device = device
        self.fb: Optional[str] = None

    @torch.no_grad()
    def get_train_tuple(
        self,
        x_pairs: torch.Tensor,
        fb: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z0, z1 = x_pairs[:, 0].to(self.device), x_pairs[:, 1].to(self.device)
        t = torch.rand((z1.shape[0], 1), device=self.device) * (1 - 2 * self.eps) + self.eps
        z_t = t * z1 + (1.0 - t) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sigma * torch.sqrt(t * (1.0 - t)) * z
        if fb == "f":
            target = z1 - z0
            target = target - self.sigma * torch.sqrt(t / (1.0 - t)) * z
        else:
            target = -(z1 - z0)
            target = target - self.sigma * torch.sqrt((1.0 - t) / t) * z
        return z_t.cpu(), t.cpu(), target.cpu()

    @torch.no_grad()
    def generate_new_dataset(
        self,
        x_pairs: torch.Tensor,
        prev_model: Optional["DSBM"],
        fb: str,
        first_it: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert fb in {"f", "b"}
        if prev_model is None:
            assert first_it and fb == "b"
            zstart = x_pairs[:, 0]
            if self.first_coupling == "ref":
                zend = zstart + torch.randn_like(zstart) * self.sigma
            elif self.first_coupling == "ind":
                zend = x_pairs[:, 1].clone()
                perm = torch.randperm(len(zend))
                zend = zend[perm]
            else:
                raise NotImplementedError
            z0, z1 = zstart, zend
        else:
            assert not first_it
            if prev_model.fb == "f":
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
            if prev_model.fb == "f":
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        return z0.cpu(), z1.cpu()

    @torch.no_grad()
    def sample_sde(
        self,
        zstart: torch.Tensor,
        fb: str = "f",
        steps: Optional[int] = None,
    ) -> list[torch.Tensor]:
        assert fb in {"f", "b"}
        N = steps or self.N
        dt = 1.0 / N
        traj = []
        z = zstart.to(self.device).detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        ts = np.arange(N, dtype=np.float32) / N
        if fb == "b":
            ts = 1 - ts
        for i in range(N):
            t = torch.full((batchsize, 1), ts[i], device=self.device)
            pred = self.net_dict[fb](z, t)
            z = z.detach().clone() + pred * dt
            z = z + self.sigma * torch.randn_like(z) * np.sqrt(dt)
            traj.append(z.detach().clone())
        return [state.cpu() for state in traj]


class RectifiedFlow(nn.Module):
    """Rectified Flow / Flow Matching solver."""

    def __init__(
        self,
        net: nn.Module,
        num_steps: int,
        sigma: Optional[float],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.net = net
        self.N = num_steps
        self.sigma = sigma or 0.0
        self.device = device
        self.fb: Optional[str] = None

    @torch.no_grad()
    def get_train_tuple(
        self,
        x_pairs: torch.Tensor,
        fb: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z0, z1 = x_pairs[:, 0].to(self.device), x_pairs[:, 1].to(self.device)
        t = torch.rand((z1.shape[0], 1), device=self.device)
        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0
        return z_t.cpu(), t.cpu(), target.cpu()

    @torch.no_grad()
    def generate_new_dataset(
        self,
        x_pairs: torch.Tensor,
        prev_model: Optional["RectifiedFlow"],
        fb: str,
        first_it: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_model is None:
            assert first_it
            z0 = x_pairs[:, 0]
            perm = torch.randperm(len(x_pairs))
            z1 = x_pairs[perm, 1]
        else:
            assert not first_it
            if prev_model.fb == "f":
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            zend = prev_model.sample_ode(zstart=zstart, fb=prev_model.fb)[-1]
            if prev_model.fb == "f":
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        return z0.cpu(), z1.cpu()

    @torch.no_grad()
    def sample_ode(
        self,
        zstart: torch.Tensor,
        fb: str = "f",
        steps: Optional[int] = None,
    ) -> list[torch.Tensor]:
        assert fb in {"f", "b"}
        N = steps or self.N
        dt = 1.0 / N
        traj = []
        z = zstart.to(self.device).detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        ts = np.arange(N, dtype=np.float32) / N
        if fb == "b":
            ts = 1 - ts
        sign = 1 if fb == "f" else -1
        for i in range(N):
            t = torch.full((batchsize, 1), ts[i], device=self.device)
            pred = sign * self.net(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return [state.cpu() for state in traj]


def _to_device(tensors: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in tensors)


def train_dsb_ipf(
    model: DSB,
    x_pairs: torch.Tensor,
    batch_size: int,
    inner_iters: int,
    learning_rate: float,
    fb: str,
    first_it: bool,
) -> Tuple[DSB, list[float]]:
    optimizer = torch.optim.Adam(model.net_dict[fb].parameters(), lr=learning_rate)
    loss_curve: list[float] = []
    z_ts, ts, targets = model.generate_new_dataset_and_train_tuple(x_pairs, fb=fb, first_it=first_it)
    dl = DataLoader(
        TensorDataset(z_ts, ts, targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    iterator = iter(dl)
    for _ in tqdm(range(inner_iters), desc=f"DSB {fb}"):
        try:
            batch = next(iterator)
        except StopIteration:
            z_ts, ts, targets = model.generate_new_dataset_and_train_tuple(x_pairs, fb=fb, first_it=first_it)
            dl = DataLoader(
                TensorDataset(z_ts, ts, targets),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            iterator = iter(dl)
            batch = next(iterator)
        z_t, t, target = _to_device(batch, model.device)
        optimizer.zero_grad()
        pred = model.net_dict[fb](z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net_dict[fb].parameters(), max_norm=1.0)
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu()))
    return model, loss_curve


def train_dsbm(
    model: DSBM,
    x_pairs: torch.Tensor,
    batch_size: int,
    inner_iters: int,
    learning_rate: float,
    fb: str,
    first_it: bool,
    prev_model: Optional[DSBM] = None,
) -> Tuple[DSBM, list[float]]:
    optimizer = torch.optim.Adam(model.net_dict[fb].parameters(), lr=learning_rate)
    loss_curve: list[float] = []
    z0, z1 = model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)
    dl = DataLoader(
        TensorDataset(z0, z1),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    iterator = iter(dl)
    for _ in tqdm(range(inner_iters), desc=f"DSBM {fb}"):
        try:
            batch = next(iterator)
        except StopIteration:
            z0, z1 = model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)
            dl = DataLoader(
                TensorDataset(z0, z1),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            iterator = iter(dl)
            batch = next(iterator)
        stacked = torch.stack(batch, dim=1)
        z_t, t, target = model.get_train_tuple(stacked, fb=fb)
        z_t, t, target = _to_device((z_t, t, target), model.device)
        optimizer.zero_grad()
        pred = model.net_dict[fb](z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net_dict[fb].parameters(), max_norm=1.0)
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu()))
    return model, loss_curve


def train_flow_model(
    model: RectifiedFlow,
    x_pairs: torch.Tensor,
    batch_size: int,
    inner_iters: int,
    learning_rate: float,
    fb: str,
    first_it: bool,
    prev_model: Optional[RectifiedFlow] = None,
) -> Tuple[RectifiedFlow, list[float]]:
    optimizer = torch.optim.Adam(model.net.parameters(), lr=learning_rate)
    loss_curve: list[float] = []
    z0, z1 = model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)
    dl = DataLoader(
        TensorDataset(z0, z1),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    iterator = iter(dl)
    for _ in tqdm(range(inner_iters), desc=f"{model.__class__.__name__} {fb}"):
        try:
            batch = next(iterator)
        except StopIteration:
            z0, z1 = model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)
            dl = DataLoader(
                TensorDataset(z0, z1),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            iterator = iter(dl)
            batch = next(iterator)
        stacked = torch.stack(batch, dim=1)
        z_t, t, target = model.get_train_tuple(stacked, fb=fb)
        z_t, t, target = _to_device((z_t, t, target), model.device)
        optimizer.zero_grad()
        pred = model.net(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=1.0)
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu()))
    return model, loss_curve
