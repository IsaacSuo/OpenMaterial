"""
PBR (Physically Based Rendering) utilities for 2DGS-PBR

Implements:
- Environment lighting (HDR environment maps)
- Cook-Torrance BRDF
- Screen-space PBR shading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class EnvironmentLight(nn.Module):
    """
    Environment lighting using HDR environment maps.
    Supports loading from .hdr/.exr files or creating from spherical harmonics.
    """

    def __init__(self, env_map_path: str = None, resolution: int = 512):
        """
        Args:
            env_map_path: Path to HDR environment map (.hdr, .exr, or .png)
            resolution: Resolution for the environment map (width = 2*height)
        """
        super().__init__()
        self.resolution = resolution

        if env_map_path is not None and os.path.exists(env_map_path):
            self.env_map = self._load_env_map(env_map_path)
        else:
            # Default: simple ambient light (gray environment)
            self.env_map = nn.Parameter(
                torch.ones(3, resolution, resolution * 2) * 0.5,
                requires_grad=True
            )

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
    Compute PBR shading with environment lighting (importance sampling).

    This is a simplified version using reflection direction sampling.
    For more accurate results, use pre-filtered environment maps.

    Args:
        albedo: Base color [H, W, 3]
        roughness: Roughness [H, W, 1]
        metallic: Metallic [H, W, 1]
        normal: Surface normal [H, W, 3]
        view_dir: View direction [H, W, 3] (pointing to camera)
        env_light: EnvironmentLight module
        num_samples: Number of samples for importance sampling

    Returns:
        Shaded color [H, W, 3]
    """
    H, W = albedo.shape[:2]
    device = albedo.device

    # Reflection direction for specular
    reflect_dir = 2.0 * torch.sum(normal * view_dir, dim=-1, keepdim=True) * normal - view_dir
    reflect_dir = F.normalize(reflect_dir, dim=-1)

    # Sample environment map at reflection direction for specular
    specular_color = env_light.sample(reflect_dir)

    # Sample environment map at normal direction for diffuse (irradiance)
    diffuse_irradiance = env_light.sample(normal)

    # Base reflectivity
    f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)

    # Simplified Fresnel for environment
    n_dot_v = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), min=0.001)
    F_env = fresnel_schlick(n_dot_v, f0)

    # Energy conservation
    kS = F_env
    kD = (1.0 - kS) * (1.0 - metallic)

    # Combine diffuse and specular
    diffuse = kD * albedo * diffuse_irradiance
    specular = kS * specular_color * (1.0 - roughness)  # Rough surfaces = less sharp reflection

    color = diffuse + specular

    return color


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

    # Compute view direction for each pixel
    # This is a simplified version - in practice, you'd compute this from depth and camera intrinsics
    # For now, we use the camera center direction
    # view_dir points FROM surface TO camera
    if gbuffer_depth is not None:
        # TODO: Compute actual view directions from depth and camera intrinsics
        pass

    # Simplified: assume view direction is towards camera center
    # This is an approximation that works reasonably for rendering
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

    # Clamp to valid range
    shaded = torch.clamp(shaded, 0.0, 1.0)

    return shaded
