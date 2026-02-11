"""
Light probe networks for 2DGS-PBR.

This module provides a position+direction -> radiance model that can replace (or augment)
the far-field EnvironmentLight env_map for PBR shading.

Backends:
  - "tcnn": TinyCUDA-NN HashGrid + small MLP (preferred, requires tinycudann)
  - "mlp": Pure PyTorch fallback (not HashGrid; for environments without tinycudann)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LightProbeConfig:
    backend: str = "tcnn"  # "tcnn" | "mlp"
    # AABB used to normalize world positions into [0,1] for the encoding.
    aabb_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    aabb_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # tcnn HashGrid params (defaults follow common instant-ngp patterns).
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 1.5

    # Direction encoding.
    dir_encoding: str = "sh"  # "sh" | "fourier"
    sh_degree: int = 4
    fourier_n_frequencies: int = 6

    # Decoder MLP.
    hidden_dim: int = 64
    n_hidden_layers: int = 2

    # Output mapping.
    output_activation: str = "softplus"  # "softplus" | "exp" | "none"
    output_softplus_beta: float = 1.0


def _normalize_positions_to_01(xyz: torch.Tensor, aabb_min: torch.Tensor, aabb_max: torch.Tensor) -> torch.Tensor:
    # xyz: [...,3]
    denom = (aabb_max - aabb_min).clamp_min(1e-8)
    return (xyz - aabb_min) / denom


def _fourier_encode_dir(d: torch.Tensor, n_freq: int) -> torch.Tensor:
    # d: [N,3] in [-1,1], unit length preferred.
    d = torch.clamp(d, -1.0, 1.0)
    freqs = (2.0 ** torch.arange(n_freq, device=d.device, dtype=d.dtype)) * torch.pi
    # [N, 3, n_freq]
    x = d.unsqueeze(-1) * freqs.view(1, 1, -1)
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [N,3,2*n_freq]
    return enc.reshape(d.shape[0], -1)


class _MLPLightProbe(nn.Module):
    def __init__(self, cfg: LightProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("aabb_min", torch.tensor(cfg.aabb_min, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(cfg.aabb_max, dtype=torch.float32))

        in_dim = 3  # normalized xyz
        if cfg.dir_encoding == "fourier":
            dir_dim = 3 * 2 * int(cfg.fourier_n_frequencies)
        else:
            # SH is not implemented in the pure torch fallback; use raw dir.
            dir_dim = 3

        layers: list[nn.Module] = []
        last = in_dim + dir_dim
        for _ in range(int(cfg.n_hidden_layers)):
            layers.append(nn.Linear(last, int(cfg.hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            last = int(cfg.hidden_dim)
        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz_world: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
        # xyz_world: [N,3], dirs_world: [N,3]
        xyz01 = _normalize_positions_to_01(
            xyz_world,
            aabb_min=self.aabb_min.to(xyz_world.dtype),
            aabb_max=self.aabb_max.to(xyz_world.dtype),
        )
        xyz01 = torch.clamp(xyz01, 0.0, 1.0)

        if self.cfg.dir_encoding == "fourier":
            denc = _fourier_encode_dir(F.normalize(dirs_world, dim=-1), int(self.cfg.fourier_n_frequencies))
        else:
            denc = F.normalize(dirs_world, dim=-1)

        x = torch.cat([xyz01, denc], dim=-1)
        out = self.net(x)
        return out


class TCNNLightProbe(nn.Module):
    """
    TinyCUDA-NN HashGrid light probe (xyz + dir -> RGB radiance).
    """

    def __init__(self, cfg: LightProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("aabb_min", torch.tensor(cfg.aabb_min, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(cfg.aabb_max, dtype=torch.float32))

        try:
            import tinycudann as tcnn  # type: ignore
        except Exception as e:
            raise ImportError(
                "tinycudann is required for backend='tcnn'. "
                "Install TinyCUDA-NN python bindings (package name often 'tinycudann') "
                "or use backend='mlp'."
            ) from e

        pos_enc_cfg = {
            "otype": "HashGrid",
            "n_levels": int(cfg.n_levels),
            "n_features_per_level": int(cfg.n_features_per_level),
            "log2_hashmap_size": int(cfg.log2_hashmap_size),
            "base_resolution": int(cfg.base_resolution),
            "per_level_scale": float(cfg.per_level_scale),
        }
        self.pos_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=pos_enc_cfg)

        if cfg.dir_encoding == "sh":
            dir_enc_cfg = {"otype": "SphericalHarmonics", "degree": int(cfg.sh_degree)}
            self.dir_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=dir_enc_cfg)
        elif cfg.dir_encoding == "fourier":
            # Keep direction encoding in torch for flexibility; this is small.
            self.dir_encoding = None
        else:
            raise ValueError(f"Unsupported dir_encoding: {cfg.dir_encoding}")

        pos_dim = self.pos_encoding.n_output_dims
        if self.dir_encoding is not None:
            dir_dim = self.dir_encoding.n_output_dims
        elif cfg.dir_encoding == "fourier":
            dir_dim = 3 * 2 * int(cfg.fourier_n_frequencies)
        else:
            dir_dim = 3

        mlp_cfg = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(cfg.hidden_dim),
            "n_hidden_layers": int(cfg.n_hidden_layers),
        }
        self.decoder = tcnn.Network(n_input_dims=int(pos_dim + dir_dim), n_output_dims=3, network_config=mlp_cfg)

    def forward(self, xyz_world: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
        # xyz_world, dirs_world: [N,3]
        aabb_min = self.aabb_min.to(device=xyz_world.device, dtype=xyz_world.dtype)
        aabb_max = self.aabb_max.to(device=xyz_world.device, dtype=xyz_world.dtype)
        xyz01 = _normalize_positions_to_01(xyz_world, aabb_min=aabb_min, aabb_max=aabb_max)
        xyz01 = torch.clamp(xyz01, 0.0, 1.0)

        d = F.normalize(dirs_world, dim=-1)

        f_pos = self.pos_encoding(xyz01)
        if self.dir_encoding is not None:
            f_dir = self.dir_encoding(d)
        else:
            f_dir = _fourier_encode_dir(d, int(self.cfg.fourier_n_frequencies))
        x = torch.cat([f_pos, f_dir], dim=-1)
        out = self.decoder(x)
        return out


class LightProbe(nn.Module):
    """
    Wrapper that exposes a stable API and supports optional TinyCUDA-NN.
    """

    def __init__(self, cfg: LightProbeConfig):
        super().__init__()
        self.cfg = cfg
        backend = str(cfg.backend).lower().strip()
        if backend not in ("tcnn", "mlp"):
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend

        if backend == "tcnn":
            try:
                self.impl = TCNNLightProbe(cfg)
            except ImportError as e:
                # Fallback to MLP so code can still run in CPU-only / no-TCNN envs.
                print(f"[LightProbe] Warning: {e}. Falling back to backend='mlp'.")
                self.backend = "mlp"
                self.impl = _MLPLightProbe(cfg)
        else:
            self.impl = _MLPLightProbe(cfg)

    def forward(self, xyz_world: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
        return self.impl(xyz_world, dirs_world)

    def radiance(self, xyz_world: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
        """
        Returns HDR radiance in linear RGB.
        """
        out = self.forward(xyz_world, dirs_world)
        act = str(self.cfg.output_activation).lower().strip()
        if act == "softplus":
            return F.softplus(out, beta=float(self.cfg.output_softplus_beta))
        if act == "exp":
            return torch.exp(out)
        if act == "none":
            return out
        raise ValueError(f"Unsupported output_activation: {self.cfg.output_activation}")

