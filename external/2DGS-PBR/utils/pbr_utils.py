"""
PBR (Physically Based Rendering) utilities for 2DGS-PBR

Implements:
- Environment lighting (HDR environment maps)
- Ground plane with texture (for finite-depth backgrounds)
- Cook-Torrance BRDF
- Screen-space PBR shading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Optional, Union, Tuple


def tonemap_reinhard(x: torch.Tensor) -> torch.Tensor:
    """
    Simple Reinhard tone mapping (differentiable).
    Maps HDR radiance to SDR display range: x -> x / (1 + x), with x clamped to >=0.
    """
    x = torch.clamp_min(x, 0.0)
    return x / (1.0 + x)


class EnvironmentLight(nn.Module):
    """
    Environment lighting using HDR environment maps.
    Supports loading from .hdr/.exr files or creating from spherical harmonics.

    Includes solid-angle weighting to correct for equirectangular projection distortion.
    The weight w(v) = sin(π·v) compensates for pole singularities where pixel density
    is non-uniform across the sphere.
    """

    def __init__(self, env_map_path: str = None, resolution: int = 512, num_mip_levels: int = 5):
        """
        Args:
            env_map_path: Path to HDR environment map (.hdr, .exr, or .png)
            resolution: Resolution for the environment map (width = 2*height)
            num_mip_levels: Number of mipmap levels for roughness-based prefiltering
        """
        super().__init__()
        self.resolution = resolution
        self.num_mip_levels = num_mip_levels

        if env_map_path is not None and os.path.exists(env_map_path):
            self.env_map = self._load_env_map(env_map_path)
        else:
            # Default: simple ambient light (gray environment)
            self.env_map = nn.Parameter(
                torch.ones(3, resolution, resolution * 2) * 0.5,
                requires_grad=True
            )

        # Solid angle weight: w(v) = sin(θ) where θ = π·v, v ∈ [0, 1]
        # This corrects for the Jacobian of equirectangular projection
        # dω = sin(θ) dθ dφ, but pixels are uniform in (θ, φ) space
        self._init_solid_angle_weight()

        # Precompute gaussian blur kernels for mipmap generation
        self._init_blur_kernels()

    def _init_solid_angle_weight(self):
        """
        Initialize solid angle weight map for equirectangular projection.

        Mathematical basis:
        - Spherical surface element: dA = sin(θ) dθ dφ
        - In pixel space (v ∈ [0,1] for latitude): θ = π·v
        - Weight per pixel row: w(v) = sin(π·v)
        - This ensures equal contribution per unit solid angle
        """
        H = self.resolution
        W = self.resolution * 2

        # v ranges from 0 (north pole) to 1 (south pole)
        # θ = π·v ranges from 0 to π
        v = torch.linspace(0, 1, H)
        theta = np.pi * v

        # sin(θ) weight - note: sin(0) = sin(π) = 0 at poles
        # Add small epsilon to avoid division by zero in weighted operations
        solid_angle_weight = torch.sin(theta).clamp(min=1e-6)

        # Expand to [1, H, W] for broadcasting with env_map [3, H, W]
        solid_angle_weight = solid_angle_weight.view(1, H, 1).expand(1, H, W)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('solid_angle_weight', solid_angle_weight)

        # Precompute normalization factor for weighted mean
        self.register_buffer('weight_sum', solid_angle_weight.sum())

    def _init_blur_kernels(self):
        """
        Initialize Gaussian blur kernels for differentiable mipmap generation.

        Each mip level uses progressively larger blur to approximate the BRDF lobe convolution.
        The kernel sizes are chosen to approximate the GGX lobe width at each roughness level.
        """
        # Kernel sizes for each mip level (must be odd)
        # Level 0: no blur (roughness ≈ 0, mirror)
        # Level N: maximum blur (roughness ≈ 1, diffuse)
        self.blur_kernel_sizes = []
        self.blur_sigmas = []

        for level in range(self.num_mip_levels):
            if level == 0:
                # Level 0 is the original image, no blur needed
                self.blur_kernel_sizes.append(1)
                self.blur_sigmas.append(0.0)
            else:
                # Progressively larger blur
                # Sigma increases exponentially to match GGX lobe spreading
                # At roughness=1, lobe covers ~hemisphere, so large blur
                sigma = 2.0 ** level  # 2, 4, 8, 16, ...
                kernel_size = int(sigma * 4) | 1  # Ensure odd, cover ~4 sigma
                kernel_size = min(kernel_size, 63)  # Cap kernel size
                self.blur_kernel_sizes.append(kernel_size)
                self.blur_sigmas.append(sigma)

    def _gaussian_blur(self, img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian blur to an image. Fully differentiable.

        Args:
            img: [C, H, W] or [B, C, H, W] input image
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian

        Returns:
            Blurred image with same shape as input
        """
        if kernel_size <= 1 or sigma <= 0:
            return img

        # Add batch dimension if needed
        squeeze_batch = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = img.shape

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable convolution for efficiency
        # Horizontal pass
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
        # Handle horizontal wrapping for equirectangular map
        pad_h = kernel_size // 2
        img_padded = F.pad(img, (pad_h, pad_h, 0, 0), mode='circular')
        img = F.conv2d(img_padded, kernel_h, groups=C)

        # Vertical pass
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
        pad_v = kernel_size // 2
        img_padded = F.pad(img, (0, 0, pad_v, pad_v), mode='reflect')
        img = F.conv2d(img_padded, kernel_v, groups=C)

        if squeeze_batch:
            img = img.squeeze(0)

        return img

    def _build_mipmaps(self) -> list:
        """
        Dynamically build mipmap pyramid from the current env_map.

        This is fully differentiable - gradients flow back to self.env_map.
        Called on each forward pass to ensure mipmaps reflect current parameters.

        Returns:
            List of [3, H, W] tensors, one per mip level (0 = original, N = most blurred)
        """
        mipmaps = [self.env_map]  # Level 0 is original

        current = self.env_map
        for level in range(1, self.num_mip_levels):
            kernel_size = self.blur_kernel_sizes[level]
            sigma = self.blur_sigmas[level]
            blurred = self._gaussian_blur(current, kernel_size, sigma)
            mipmaps.append(blurred)
            # Note: We blur from original each time, not cascaded
            # This gives more control over each level's blur amount

        return mipmaps

    def sample_prefiltered(self, directions: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """
        Sample environment map with roughness-dependent prefiltering (Split-Sum approximation).

        This implements differentiable mipmap-based sampling where:
        - roughness ≈ 0 → sample from sharp (level 0) environment
        - roughness ≈ 1 → sample from blurred (level N) environment

        The operation is fully differentiable, allowing gradients to flow to:
        1. The environment map parameters (self.env_map)
        2. The roughness values (for material optimization)

        Args:
            directions: [N, 3] or [H, W, 3] normalized direction vectors
            roughness: [N, 1] or [H, W, 1] roughness values in [0, 1]

        Returns:
            colors: Same shape as directions with 3 channels for RGB
        """
        original_shape = directions.shape[:-1]
        directions = directions.reshape(-1, 3)
        roughness = roughness.reshape(-1, 1)

        # Build mipmaps dynamically (differentiable)
        mipmaps = self._build_mipmaps()

        # Convert direction to UV coordinates
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        theta = torch.atan2(x, z)
        u = (theta / (2 * np.pi) + 0.5) % 1.0
        phi = torch.acos(torch.clamp(y, -1.0, 1.0))
        v = phi / np.pi

        grid_u = u * 2 - 1
        grid_v = v * 2 - 1
        grid = torch.stack([grid_u, grid_v], dim=-1)  # [N, 2]

        # Compute LOD (Level of Detail) from roughness
        # Roughness 0 -> level 0, roughness 1 -> level (num_mip_levels - 1)
        max_level = self.num_mip_levels - 1
        lod = roughness.squeeze(-1) * max_level  # [N]
        lod = torch.clamp(lod, 0, max_level)

        lod_floor = torch.floor(lod).long()  # [N]
        lod_ceil = torch.clamp(lod_floor + 1, 0, max_level).long()  # [N]
        lod_frac = lod - lod_floor.float()  # [N] interpolation weight

        # Sample from all mip levels (we'll select per-pixel)
        # This is less efficient but ensures differentiability
        N = directions.shape[0]
        grid_expanded = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

        sampled_levels = []
        for mip in mipmaps:
            mip_batch = mip.unsqueeze(0)  # [1, 3, H, W]
            sampled = F.grid_sample(
                mip_batch, grid_expanded,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # [1, 3, 1, N]
            sampled_levels.append(sampled.squeeze(0).squeeze(1).permute(1, 0))  # [N, 3]

        # Stack all levels: [num_levels, N, 3]
        all_levels = torch.stack(sampled_levels, dim=0)

        # Gather floor and ceil level samples for each pixel
        # Create indices for gather: [N]
        batch_idx = torch.arange(N, device=directions.device)

        # Get floor level samples
        val_floor = all_levels[lod_floor, batch_idx, :]  # [N, 3]
        val_ceil = all_levels[lod_ceil, batch_idx, :]  # [N, 3]

        # Trilinear interpolation
        lod_frac_expanded = lod_frac.unsqueeze(-1)  # [N, 1]
        result = val_floor * (1 - lod_frac_expanded) + val_ceil * lod_frac_expanded  # [N, 3]

        return result.reshape(*original_shape, 3)

    def get_weighted_env_map(self) -> torch.Tensor:
        """
        Returns environment map weighted by solid angle.
        Useful for computing weighted statistics or losses.

        Returns:
            weighted_map: [3, H, W] environment map scaled by sin(θ)
        """
        return self.env_map * self.solid_angle_weight

    def tv_loss_weighted(self) -> torch.Tensor:
        """
        Compute Total Variation loss with solid angle weighting.

        Standard TV loss over-penalizes smoothness at poles due to high pixel density.
        Weighting by sin(θ) ensures uniform regularization per unit solid angle.

        Returns:
            weighted_tv: Scalar TV loss with solid angle correction
        """
        # Horizontal gradient (along φ direction)
        # At poles, horizontal neighbors are nearly identical in 3D, so reduce weight
        diff_h = self.env_map[:, :, 1:] - self.env_map[:, :, :-1]  # [3, H, W-1]
        weight_h = self.solid_angle_weight[:, :, :-1]  # [1, H, W-1]
        tv_h = (diff_h.abs() * weight_h).sum()

        # Vertical gradient (along θ direction)
        diff_v = self.env_map[:, 1:, :] - self.env_map[:, :-1, :]  # [3, H-1, W]
        # Use average weight between adjacent rows
        weight_v = (self.solid_angle_weight[:, 1:, :] + self.solid_angle_weight[:, :-1, :]) / 2
        tv_v = (diff_v.abs() * weight_v).sum()

        # Normalize by total weight to make loss scale-invariant
        return (tv_h + tv_v) / self.weight_sum

    def smoothness_loss_weighted(self) -> torch.Tensor:
        """
        Compute L2 smoothness loss (Laplacian) with solid angle weighting.

        Returns:
            weighted_smooth: Scalar smoothness loss
        """
        # Second derivative approximation using Laplacian kernel
        diff_h = self.env_map[:, :, 2:] - 2 * self.env_map[:, :, 1:-1] + self.env_map[:, :, :-2]
        diff_v = self.env_map[:, 2:, :] - 2 * self.env_map[:, 1:-1, :] + self.env_map[:, :-2, :]

        weight_h = self.solid_angle_weight[:, :, 1:-1]
        weight_v = self.solid_angle_weight[:, 1:-1, :]

        smooth_h = ((diff_h ** 2) * weight_h).sum()
        smooth_v = ((diff_v ** 2) * weight_v).sum()

        return (smooth_h + smooth_v) / self.weight_sum

    def register_gradient_scaling_hook(self):
        """
        Register a backward hook to scale gradients by solid angle weight.

        This prevents pole regions from dominating gradient updates due to
        their artificially high pixel density in equirectangular projection.

        Call this after optimizer setup if you want gradient-level correction.
        """
        def _scale_gradient(grad):
            # Scale gradient inversely by solid angle weight
            # Poles (low weight) get reduced gradient; equator (high weight) preserved
            return grad * self.solid_angle_weight

        self.env_map.register_hook(_scale_gradient)
        print("[EnvironmentLight] Registered solid-angle gradient scaling hook")

    def get_effective_resolution_stats(self) -> dict:
        """
        Compute statistics about effective resolution across the sphere.
        Useful for debugging and understanding the projection distortion.

        Returns:
            dict with statistics about sampling density
        """
        weight = self.solid_angle_weight.squeeze()
        return {
            'min_weight': weight.min().item(),
            'max_weight': weight.max().item(),
            'mean_weight': weight.mean().item(),
            'equator_weight': weight[self.resolution // 2, 0].item(),
            'pole_weight': weight[0, 0].item(),
            'effective_pixels': (weight.sum() / weight.max()).item(),
        }

    def _load_env_map(self, path: str) -> nn.Parameter:
        """Load environment map from file"""
        import imageio

        # Load HDR image
        if path.endswith('.hdr') or path.endswith('.exr'):
            try:
                import imageio.v2 as iio
                env_img = iio.imread(path, format='HDR-FI')
            except:
                env_img = imageio.imread(path)
        else:
            env_img = imageio.imread(path).astype(np.float32) / 255.0

        # Convert to torch tensor [C, H, W]
        env_tensor = torch.from_numpy(env_img).permute(2, 0, 1).float()

        # Resize if needed
        if env_tensor.shape[1] != self.resolution:
            env_tensor = F.interpolate(
                env_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution * 2),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return nn.Parameter(env_tensor, requires_grad=True)

    def sample(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Sample environment map given world-space directions.

        Args:
            directions: [N, 3] or [H, W, 3] normalized direction vectors

        Returns:
            colors: Same shape as input with 3 channels for RGB
        """
        original_shape = directions.shape[:-1]
        directions = directions.reshape(-1, 3)

        # Convert direction to spherical coordinates
        # theta: azimuth angle [0, 2*pi]
        # phi: elevation angle [0, pi]
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

        # Azimuth (longitude): atan2(x, z) -> [-pi, pi] -> [0, 1]
        theta = torch.atan2(x, z)
        u = (theta / (2 * np.pi) + 0.5) % 1.0

        # Elevation (latitude): acos(y) -> [0, pi] -> [0, 1]
        phi = torch.acos(torch.clamp(y, -1.0, 1.0))
        v = phi / np.pi

        # Sample from environment map using grid_sample
        # grid_sample expects coordinates in [-1, 1]
        grid_u = u * 2 - 1  # [0, 1] -> [-1, 1]
        grid_v = v * 2 - 1  # [0, 1] -> [-1, 1]

        grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0).unsqueeze(0)
        # grid shape: [1, 1, N, 2]

        env_map = self.env_map.unsqueeze(0)  # [1, 3, H, W]
        sampled = F.grid_sample(
            env_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        # sampled shape: [1, 3, 1, N]

        colors = sampled.squeeze(0).squeeze(1).permute(1, 0)  # [N, 3]
        return colors.reshape(*original_shape, 3)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Alias for sample()"""
        return self.sample(directions)


class GroundPlane(nn.Module):
    """
    Ground plane with texture for finite-depth background rendering.

    This complements EnvironmentLight by providing a way to render finite-geometry
    backgrounds (like a checkerboard floor) that have correct parallax.

    For each pixel:
    - If the camera ray intersects the ground plane → sample ground texture
    - Otherwise → fall back to environment map (sky)

    The ground plane is defined by:
    - Plane equation: n·x + d = 0
    - Basis vectors: u, v (tangent directions on the plane)
    - Origin: p0 (a point on the plane)
    - UV bounds: the texture covers [uv_min, uv_max] in plane coordinates
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        texture_path: Optional[str] = None,
    ):
        """
        Args:
            json_path: Path to ground_plane.json (plane parameters)
            texture_path: Path to ground_texture.png
        """
        super().__init__()

        # Default: horizontal plane at y=0, facing up
        self.register_buffer('plane_normal', torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer('plane_d', torch.tensor(0.0))
        self.register_buffer('plane_origin', torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer('plane_u', torch.tensor([1.0, 0.0, 0.0]))
        self.register_buffer('plane_v', torch.tensor([0.0, 0.0, 1.0]))
        self.register_buffer('uv_min', torch.tensor([0.0, 0.0]))
        self.register_buffer('uv_max', torch.tensor([1.0, 1.0]))

        # Default texture: gray
        self.ground_texture = nn.Parameter(
            torch.ones(3, 64, 64) * 0.5,
            requires_grad=False  # Typically frozen
        )

        if json_path is not None and os.path.exists(json_path):
            self._load_plane_params(json_path)

        if texture_path is not None and os.path.exists(texture_path):
            self._load_texture(texture_path)

    def _load_plane_params(self, json_path: str):
        """Load plane parameters from ground_plane.json"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        plane = data['plane']
        basis = data['basis']
        uv_bounds = data['uv_bounds']

        # Plane equation: n·x + d = 0
        self.plane_normal.data = torch.tensor(plane['n'], dtype=torch.float32)
        self.plane_d.data = torch.tensor(plane['d'], dtype=torch.float32)

        # Plane basis
        self.plane_origin.data = torch.tensor(basis['p0'], dtype=torch.float32)
        self.plane_u.data = torch.tensor(basis['u'], dtype=torch.float32)
        self.plane_v.data = torch.tensor(basis['v'], dtype=torch.float32)

        # UV bounds
        self.uv_min.data = torch.tensor(uv_bounds['min'], dtype=torch.float32)
        self.uv_max.data = torch.tensor(uv_bounds['max'], dtype=torch.float32)

        print(f"[GroundPlane] Loaded plane params from {json_path}")
        print(f"  normal={self.plane_normal.tolist()}, d={self.plane_d.item():.4f}")
        print(f"  UV bounds: [{self.uv_min.tolist()}] to [{self.uv_max.tolist()}]")

    def _load_texture(self, texture_path: str):
        """Load ground texture from image file"""
        import imageio

        img = imageio.imread(texture_path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Handle grayscale
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha

        # Convert to [C, H, W]
        tex = torch.from_numpy(img).permute(2, 0, 1).float()

        # Note: texture in the file has V flipped (top = +v_max)
        # We flip it so that V increases downward in texture space
        tex = torch.flip(tex, dims=[1])

        self.ground_texture = nn.Parameter(tex, requires_grad=False)
        print(f"[GroundPlane] Loaded texture from {texture_path}, shape={tuple(tex.shape)}")

    def ray_plane_intersect(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ray-plane intersection.

        Plane equation: n·x + d = 0
        Ray: o + t*dir
        Intersection: n·(o + t*dir) + d = 0
                      t = -(n·o + d) / (n·dir)

        Args:
            ray_origins: [..., 3] ray origins (camera position)
            ray_dirs: [..., 3] normalized ray directions

        Returns:
            t: [...] intersection distance (negative = behind camera)
            hit_mask: [...] bool, True if ray hits plane (t > 0, valid intersection)
            hit_points: [..., 3] intersection points in world space
        """
        # n·dir
        n_dot_dir = (ray_dirs * self.plane_normal).sum(dim=-1)  # [...]

        # Avoid division by zero (ray parallel to plane)
        parallel_mask = torch.abs(n_dot_dir) < 1e-6
        n_dot_dir_safe = torch.where(parallel_mask, torch.ones_like(n_dot_dir), n_dot_dir)

        # n·o + d
        n_dot_o = (ray_origins * self.plane_normal).sum(dim=-1) + self.plane_d  # [...]

        # t = -(n·o + d) / (n·dir)
        t = -n_dot_o / n_dot_dir_safe

        # Hit if t > 0 (in front of camera) and ray not parallel to plane
        hit_mask = (t > 0) & (~parallel_mask)

        # Compute hit points
        hit_points = ray_origins + t.unsqueeze(-1) * ray_dirs  # [..., 3]

        return t, hit_mask, hit_points

    def world_to_uv(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world-space points (on the plane) to UV coordinates.

        Args:
            points: [..., 3] world-space points

        Returns:
            uv: [..., 2] UV coordinates
        """
        # Relative to plane origin
        rel = points - self.plane_origin  # [..., 3]

        # Project onto plane basis
        u = (rel * self.plane_u).sum(dim=-1)  # [...]
        v = (rel * self.plane_v).sum(dim=-1)  # [...]

        return torch.stack([u, v], dim=-1)  # [..., 2]

    def sample_texture(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Sample ground texture at UV coordinates.

        Args:
            uv: [..., 2] UV coordinates in world units

        Returns:
            colors: [..., 3] RGB colors
        """
        original_shape = uv.shape[:-1]
        uv_flat = uv.reshape(-1, 2)  # [N, 2]

        # Normalize UV to [0, 1] based on bounds
        uv_range = self.uv_max - self.uv_min
        uv_normalized = (uv_flat - self.uv_min) / (uv_range + 1e-8)

        # Convert to grid_sample coordinates [-1, 1]
        grid = uv_normalized * 2.0 - 1.0  # [N, 2]

        # grid_sample expects [B, C, H, W] and grid [B, H_out, W_out, 2]
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        tex = self.ground_texture.unsqueeze(0)  # [1, 3, H, W]

        sampled = F.grid_sample(
            tex, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [1, 3, 1, N]

        colors = sampled.squeeze(0).squeeze(1).permute(1, 0)  # [N, 3]
        return colors.reshape(*original_shape, 3)

    def is_within_bounds(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Check if UV coordinates are within texture bounds.

        Args:
            uv: [..., 2] UV coordinates

        Returns:
            in_bounds: [...] bool mask
        """
        in_u = (uv[..., 0] >= self.uv_min[0]) & (uv[..., 0] <= self.uv_max[0])
        in_v = (uv[..., 1] >= self.uv_min[1]) & (uv[..., 1] <= self.uv_max[1])
        return in_u & in_v

    def sample(
        self,
        ray_dirs: torch.Tensor,
        camera_center: torch.Tensor,
        return_mask: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample ground plane color for given ray directions.

        Args:
            ray_dirs: [H, W, 3] or [N, 3] normalized ray directions (world space)
            camera_center: [3] camera position in world space
            return_mask: if True, also return hit mask

        Returns:
            colors: same shape as ray_dirs with 3 channels, RGB colors
            hit_mask: (if return_mask=True) [...] bool, True where ray hits ground
        """
        original_shape = ray_dirs.shape[:-1]
        ray_dirs_flat = ray_dirs.reshape(-1, 3)  # [N, 3]

        # Expand camera center to match rays
        # camera_center is typically [3]; reshape to [1,3] before expanding.
        ray_origins = camera_center.view(1, 3).expand(ray_dirs_flat.shape[0], 3)  # [N, 3]

        # Ray-plane intersection
        t, hit_mask, hit_points = self.ray_plane_intersect(ray_origins, ray_dirs_flat)

        # World to UV
        uv = self.world_to_uv(hit_points)  # [N, 2]

        # Check bounds
        in_bounds = self.is_within_bounds(uv)  # [N]

        # Final hit mask: intersection + within bounds
        final_hit = hit_mask & in_bounds  # [N]

        # Sample texture
        colors = self.sample_texture(uv)  # [N, 3]

        # Zero out colors for non-hits (will be replaced by env_light)
        colors = colors * final_hit.unsqueeze(-1).float()

        colors = colors.reshape(*original_shape, 3)
        final_hit = final_hit.reshape(*original_shape)

        if return_mask:
            return colors, final_hit
        return colors

    def forward(
        self,
        ray_dirs: torch.Tensor,
        camera_center: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample() with return_mask=True"""
        return self.sample(ray_dirs, camera_center, return_mask=True)


def compute_ray_directions_world_from_fov(
    image_height: int,
    image_width: int,
    fovx: float,
    fovy: float,
    world_view_transform: torch.Tensor,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Compute per-pixel camera ray directions in world space for a pinhole camera.

    Args:
        image_height: H
        image_width: W
        fovx: horizontal FoV in radians
        fovy: vertical FoV in radians
        world_view_transform: [4, 4] world-to-view transform (same convention as Camera.world_view_transform)
        device: torch device for output (defaults to world_view_transform.device)

    Returns:
        ray_dirs_world: [H, W, 3] normalized direction vectors (from camera into the scene, world space)
    """
    if device is None:
        device = world_view_transform.device

    H, W = int(image_height), int(image_width)
    tan_fovx = float(np.tan(fovx * 0.5))
    tan_fovy = float(np.tan(fovy * 0.5))
    fx = W / (2.0 * tan_fovx)
    fy = H / (2.0 * tan_fovy)
    cx = W / 2.0
    cy = H / 2.0

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy",
    )

    # Camera-space directions (z = +1)
    dirs_c = torch.stack(
        [
            (grid_x - cx) / fx,
            (grid_y - cy) / fy,
            torch.ones_like(grid_x),
        ],
        dim=-1,
    )  # [H, W, 3]

    # Convert to world space. In this repo, Camera.world_view_transform is stored transposed.
    # The equivalent w2c matrix is world_view_transform.T.
    w2c = world_view_transform.transpose(0, 1)
    c2w_R = w2c[:3, :3].T
    dirs_w = dirs_c @ c2w_R.T

    return F.normalize(dirs_w, dim=-1)


def fresnel_schlick(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """
    Fresnel-Schlick approximation.

    Args:
        cos_theta: Cosine of angle between view and half vector [N, 1] or [H, W, 1]
        f0: Base reflectivity at normal incidence [N, 3] or [H, W, 3]

    Returns:
        Fresnel term [N, 3] or [H, W, 3]
    """
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)


def distribution_ggx(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    GGX/Trowbridge-Reitz normal distribution function.

    Args:
        n_dot_h: Dot product of normal and half vector [N, 1] or [H, W, 1]
        roughness: Roughness parameter [N, 1] or [H, W, 1]

    Returns:
        Distribution value [N, 1] or [H, W, 1]
    """
    a = roughness * roughness
    a2 = a * a

    n_dot_h2 = n_dot_h * n_dot_h
    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    denom = np.pi * denom * denom

    return a2 / torch.clamp(denom, min=1e-7)


def geometry_schlick_ggx(n_dot_v: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Schlick-GGX geometry function for single direction.

    Args:
        n_dot_v: Dot product of normal and view/light [N, 1]
        roughness: Roughness parameter [N, 1]

    Returns:
        Geometry value [N, 1]
    """
    r = roughness + 1.0
    k = (r * r) / 8.0

    denom = n_dot_v * (1.0 - k) + k
    return n_dot_v / torch.clamp(denom, min=1e-7)


def geometry_smith(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Smith's geometry function using Schlick-GGX.

    Args:
        n_dot_v: Dot product of normal and view direction [N, 1]
        n_dot_l: Dot product of normal and light direction [N, 1]
        roughness: Roughness parameter [N, 1]

    Returns:
        Geometry value [N, 1]
    """
    ggx1 = geometry_schlick_ggx(n_dot_v, roughness)
    ggx2 = geometry_schlick_ggx(n_dot_l, roughness)
    return ggx1 * ggx2


def pbr_shading(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    view_dir: torch.Tensor,
    light_dir: torch.Tensor,
    light_color: torch.Tensor,
    ambient: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute PBR shading using Cook-Torrance BRDF.

    Args:
        albedo: Base color [N, 3] or [H, W, 3]
        roughness: Roughness [N, 1] or [H, W, 1]
        metallic: Metallic [N, 1] or [H, W, 1]
        normal: Surface normal [N, 3] or [H, W, 3] (normalized)
        view_dir: View direction [N, 3] or [H, W, 3] (normalized, pointing to camera)
        light_dir: Light direction [N, 3] or [H, W, 3] (normalized, pointing to light)
        light_color: Light radiance [3] or [N, 3] or [H, W, 3]
        ambient: Ambient light [3] or [N, 3] or [H, W, 3] (optional)

    Returns:
        Shaded color [N, 3] or [H, W, 3]
    """
    # Ensure proper dimensions
    if ambient is None:
        ambient = torch.zeros_like(albedo)

    # Half vector
    half_vec = F.normalize(view_dir + light_dir, dim=-1)

    # Dot products (clamped to avoid negative values)
    n_dot_v = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), min=0.001)
    n_dot_l = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), min=0.0)
    n_dot_h = torch.clamp(torch.sum(normal * half_vec, dim=-1, keepdim=True), min=0.0)
    v_dot_h = torch.clamp(torch.sum(view_dir * half_vec, dim=-1, keepdim=True), min=0.0)

    # Base reflectivity (F0)
    # Dielectrics: 0.04, Metals: use albedo
    f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)

    # Cook-Torrance BRDF
    # D: Normal Distribution Function
    D = distribution_ggx(n_dot_h, roughness)

    # G: Geometry Function
    G = geometry_smith(n_dot_v, n_dot_l, roughness)

    # Fr: Fresnel
    Fr = fresnel_schlick(v_dot_h, f0)

    # Specular term
    numerator = D * G * Fr
    denominator = 4.0 * n_dot_v * n_dot_l
    specular = numerator / torch.clamp(denominator, min=1e-7)

    # Diffuse term (energy conservation)
    kS = Fr  # Specular reflection
    kD = (1.0 - kS) * (1.0 - metallic)  # Diffuse (non-metallic)

    # Lambertian diffuse
    diffuse = kD * albedo / np.pi

    # Combine
    Lo = (diffuse + specular) * light_color * n_dot_l

    # Add ambient
    color = Lo + ambient * albedo

    return color


def pbr_shading_env(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    view_dir: torch.Tensor,
    env_light: EnvironmentLight,
    num_samples: int = 16,
) -> torch.Tensor:
    """
    Compute PBR shading with environment lighting using Split-Sum approximation.

    This implementation uses roughness-dependent prefiltered environment sampling:
    - Smooth surfaces (low roughness) → sharp reflections from Level 0
    - Rough surfaces (high roughness) → blurred reflections from higher mip levels

    The operation is fully differentiable, allowing the optimizer to learn:
    1. Correct roughness values that match observed reflection blur
    2. Environment map content that explains the observed lighting

    Args:
        albedo: Base color [H, W, 3]
        roughness: Roughness [H, W, 1]
        metallic: Metallic [H, W, 1]
        normal: Surface normal [H, W, 3]
        view_dir: View direction [H, W, 3] (pointing to camera)
        env_light: EnvironmentLight module
        num_samples: Number of samples for importance sampling (unused, kept for API compatibility)

    Returns:
        Shaded color [H, W, 3]
    """
    H, W = albedo.shape[:2]
    device = albedo.device

    # Reflection direction for specular
    reflect_dir = 2.0 * torch.sum(normal * view_dir, dim=-1, keepdim=True) * normal - view_dir
    reflect_dir = F.normalize(reflect_dir, dim=-1)

    # Sample environment map at reflection direction for specular
    # Use roughness-dependent prefiltering (Split-Sum approximation)
    # This is the key change: rough surfaces sample from blurred mip levels
    specular_color = env_light.sample_prefiltered(reflect_dir, roughness)

    # Sample environment map at normal direction for diffuse (irradiance)
    # For diffuse, we use maximum roughness (fully blurred) as approximation of irradiance
    diffuse_roughness = torch.ones_like(roughness)
    diffuse_irradiance = env_light.sample_prefiltered(normal, diffuse_roughness)

    # Base reflectivity (Fresnel at normal incidence)
    # Dielectric: F0 ≈ 0.04, Metal: F0 = albedo
    f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)

    # Fresnel term with roughness-dependent attenuation
    # Using Schlick approximation with roughness correction
    n_dot_v = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), min=0.001)
    F_env = fresnel_schlick_roughness(n_dot_v, f0, roughness)

    # Energy conservation
    kS = F_env
    kD = (1.0 - kS) * (1.0 - metallic)

    # Combine diffuse and specular
    # Note: No longer multiply specular by (1 - roughness) since blur is handled by mipmaps
    diffuse = kD * albedo * diffuse_irradiance
    specular = kS * specular_color

    color = diffuse + specular

    return color


def fresnel_schlick_roughness(cos_theta: torch.Tensor, f0: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Fresnel-Schlick approximation with roughness correction for IBL.

    At grazing angles, rough surfaces should have reduced Fresnel effect
    compared to smooth surfaces.

    Args:
        cos_theta: Cosine of angle [H, W, 1]
        f0: Base reflectivity [H, W, 3]
        roughness: Surface roughness [H, W, 1]

    Returns:
        Fresnel term [H, W, 3]
    """
    # Clamp to prevent numerical issues
    one_minus_cos = torch.clamp(1.0 - cos_theta, 0.0, 1.0)

    # Roughness attenuates the Fresnel effect at grazing angles
    # This approximates the reduced coherent reflection from rough surfaces
    return f0 + (torch.clamp(1.0 - roughness, min=f0) - f0) * torch.pow(one_minus_cos, 5.0)


def screen_space_pbr_shading(
    gbuffer_albedo: torch.Tensor,
    gbuffer_roughness: torch.Tensor,
    gbuffer_metallic: torch.Tensor,
    gbuffer_normal: torch.Tensor,
    gbuffer_depth: torch.Tensor,
    camera_center: torch.Tensor,
    camera_transform: torch.Tensor,
    env_light: EnvironmentLight = None,
    light_dir: torch.Tensor = None,
    light_color: torch.Tensor = None,
    ray_dirs_world: Optional[torch.Tensor] = None,
    clamp_output: bool = True,
) -> torch.Tensor:
    """
    Apply PBR shading in screen space using G-Buffer.

    Args:
        gbuffer_albedo: [3, H, W] albedo map
        gbuffer_roughness: [1, H, W] roughness map
        gbuffer_metallic: [1, H, W] metallic map
        gbuffer_normal: [3, H, W] world-space normal map
        gbuffer_depth: [1, H, W] depth map
        camera_center: [3] camera position in world space
        camera_transform: [4, 4] world-to-view transform
        env_light: Optional EnvironmentLight for environment lighting
        light_dir: [3] directional light direction (if no env_light)
        light_color: [3] directional light color (if no env_light)

    Returns:
        shaded_image: [3, H, W] final shaded image
    """
    device = gbuffer_albedo.device
    C, H, W = gbuffer_albedo.shape

    # Transpose to [H, W, C] for easier processing
    albedo = gbuffer_albedo.permute(1, 2, 0)  # [H, W, 3]
    roughness = gbuffer_roughness.permute(1, 2, 0)  # [H, W, 1]
    metallic = gbuffer_metallic.permute(1, 2, 0)  # [H, W, 1]
    normal = gbuffer_normal.permute(1, 2, 0)  # [H, W, 3]

    # Normalize normals
    normal = F.normalize(normal, dim=-1)

    # View direction (world space), per pixel.
    # Prefer the true pinhole ray directions if provided; for a surface point on that ray, view_dir = -ray_dir.
    if ray_dirs_world is not None:
        if ray_dirs_world.shape == (3, H, W):
            ray_dirs_world = ray_dirs_world.permute(1, 2, 0)
        if ray_dirs_world.shape != (H, W, 3):
            raise ValueError(f"ray_dirs_world must be [H,W,3] or [3,H,W], got {tuple(ray_dirs_world.shape)}")
        view_dir = F.normalize(-ray_dirs_world, dim=-1)
    else:
        # Fallback approximation: constant view direction.
        view_dir = -camera_center.view(1, 1, 3).expand(H, W, 3)
        view_dir = F.normalize(view_dir, dim=-1)

    if env_light is not None:
        # Use environment lighting
        shaded = pbr_shading_env(
            albedo, roughness, metallic, normal, view_dir, env_light
        )
    else:
        # Use directional light
        if light_dir is None:
            light_dir = torch.tensor([0.5, 1.0, 0.5], device=device)
        light_dir = F.normalize(light_dir.view(1, 1, 3).expand(H, W, 3), dim=-1)

        if light_color is None:
            light_color = torch.ones(3, device=device)
        light_color = light_color.view(1, 1, 3).expand(H, W, 3)

        ambient = torch.full((H, W, 3), 0.03, device=device)

        shaded = pbr_shading(
            albedo, roughness, metallic, normal, view_dir, light_dir, light_color, ambient
        )

    # Transpose back to [C, H, W]
    shaded = shaded.permute(2, 0, 1)

    if clamp_output:
        shaded = torch.clamp(shaded, 0.0, 1.0)

    return shaded
