#!/usr/bin/env python3
"""
Static Geometry PBR Training Script for 2DGS-PBR

This script implements a "Static Geometry" pipeline where:
1. Geometry is initialized from a dense GT point cloud (with normals).
2. XYZ and Rotation are LOCKED (not optimized).
3. Only Scale, Opacity, PBR material parameters, and environment lighting are optimized.
4. Densification (split/clone/prune) is DISABLED.
5. Supervision is on the full composite (PBR object + skybox), with masked material losses on the object.

Usage:
    python train_pbr_static.py -s <dataset_path> -m <output_path> --gt_ply <path_to_dense.ply>
"""

import os
import torch
import sys
import uuid
import numpy as np
import re
import json
from argparse import ArgumentParser, Namespace
from random import randint
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim, compute_pbr_losses
from utils.pbr_utils import (
    EnvironmentLight,
    GroundPlane,
    screen_space_pbr_shading,
    compute_ray_directions_world_from_fov,
)
from utils.profiler import SimpleProfiler
from utils.general_utils import safe_state, colormap, inverse_sigmoid
from utils.image_utils import psnr, render_net_image
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from gaussian_renderer import render, network_gui
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _sample_basic_pcd(pcd: "BasicPointCloud", num_points: int, seed: int = 0) -> "BasicPointCloud":
    from scene.gaussian_model import BasicPointCloud

    if num_points <= 0:
        return pcd
    n = int(np.asarray(pcd.points).shape[0])
    if num_points >= n:
        return pcd
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(num_points), replace=False)
    pts = np.asarray(pcd.points)[idx]
    cols = np.asarray(pcd.colors)[idx] if np.asarray(pcd.colors).shape[0] == n else np.ones_like(pts) * 0.5
    nrms = np.asarray(pcd.normals)[idx] if np.asarray(pcd.normals).shape[0] == n else np.zeros_like(pts)
    return BasicPointCloud(points=pts, colors=cols, normals=nrms)

def _filter_pcd_outside_aabb(
    pcd: "BasicPointCloud",
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    margin: float,
) -> "BasicPointCloud":
    """
    Filter point cloud to keep only points outside an expanded AABB.
    Useful to avoid initializing background/unfixed points inside the object volume.
    """
    from scene.gaussian_model import BasicPointCloud

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    aabb_min = np.asarray(aabb_min, dtype=np.float32).reshape(3)
    aabb_max = np.asarray(aabb_max, dtype=np.float32).reshape(3)
    m = float(margin)
    aabb_min_e = aabb_min - m
    aabb_max_e = aabb_max + m

    inside = np.all((pts >= aabb_min_e[None, :]) & (pts <= aabb_max_e[None, :]), axis=1)
    keep = ~inside
    if keep.sum() == 0:
        return pcd

    cols = np.asarray(pcd.colors) if np.asarray(pcd.colors).shape[0] == pts.shape[0] else np.ones_like(pts) * 0.5
    nrms = np.asarray(pcd.normals) if np.asarray(pcd.normals).shape[0] == pts.shape[0] else np.zeros_like(pts)
    return BasicPointCloud(points=pts[keep], colors=cols[keep], normals=nrms[keep])

def _init_unfixed_pcd_from_dataset(
    source_path: str,
    num_points: int,
    extent: float,
    seed: int = 0,
    exclude_aabb_min: np.ndarray | None = None,
    exclude_aabb_max: np.ndarray | None = None,
    exclude_margin: float = 0.0,
) -> "BasicPointCloud":
    """
    Initialize a point cloud for the unfixed/background Gaussians.

    Priority:
    1) Blender-style: <source_path>/points3d.ply
    2) COLMAP-style: <source_path>/sparse/0/points3D.ply
    3) Fallback: random points in a cube around the scene extent
    """
    from scene.gaussian_model import BasicPointCloud

    candidates = [
        os.path.join(source_path, "points3d.ply"),
        os.path.join(source_path, "sparse", "0", "points3D.ply"),
    ]
    for ply_path in candidates:
        if os.path.exists(ply_path):
            print(f"[Unfixed] Initializing from dataset point cloud: {ply_path}")
            pcd = fetchPly(ply_path)
            if (exclude_aabb_min is not None) and (exclude_aabb_max is not None) and (exclude_margin > 0):
                before = int(np.asarray(pcd.points).shape[0])
                pcd = _filter_pcd_outside_aabb(pcd, exclude_aabb_min, exclude_aabb_max, margin=exclude_margin)
                after = int(np.asarray(pcd.points).shape[0])
                print(f"[Unfixed] Excluded object AABB points: {before} -> {after}")
            return _sample_basic_pcd(pcd, num_points=num_points, seed=seed)

    rng = np.random.default_rng(int(seed))
    e = float(extent)
    e = max(e, 1e-4)
    # Random init; optionally reject samples inside the object AABB.
    pts = []
    target = int(num_points)
    tries = 0
    max_tries = max(10_000, target * 20)
    while (len(pts) < target) and (tries < max_tries):
        tries += 1
        p = rng.uniform(low=-e, high=e, size=(3,)).astype(np.float32)
        if (exclude_aabb_min is not None) and (exclude_aabb_max is not None) and (exclude_margin > 0):
            mn = np.asarray(exclude_aabb_min, dtype=np.float32).reshape(3) - float(exclude_margin)
            mx = np.asarray(exclude_aabb_max, dtype=np.float32).reshape(3) + float(exclude_margin)
            if np.all(p >= mn) and np.all(p <= mx):
                continue
        pts.append(p)
    if len(pts) < target:
        # Fallback: accept remaining points without rejection to avoid empty init.
        extra = rng.uniform(low=-e, high=e, size=(target - len(pts), 3)).astype(np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        pts = np.concatenate([pts, extra], axis=0) if pts.size else extra
    else:
        pts = np.asarray(pts, dtype=np.float32)
    cols = np.ones_like(pts, dtype=np.float32) * 0.5
    nrms = np.zeros_like(pts, dtype=np.float32)
    print(f"[Unfixed] Initializing randomly: N={int(num_points)}, extent={e:.4f}")
    return BasicPointCloud(points=pts, colors=cols, normals=nrms)

def _set_gaussians_trainable(gaussians: GaussianModel, trainable: bool, freeze_geometry: bool):
    """
    Toggle gradients for Gaussian parameters in this training loop.

    Args:
        gaussians: GaussianModel
        trainable: whether to enable grads for learnable params
        freeze_geometry: if True, force xyz/rotation to be frozen regardless of trainable
    """
    if freeze_geometry:
        if hasattr(gaussians, "_xyz") and isinstance(gaussians._xyz, torch.nn.Parameter):
            gaussians._xyz.requires_grad_(False)
        if hasattr(gaussians, "_rotation") and isinstance(gaussians._rotation, torch.nn.Parameter):
            gaussians._rotation.requires_grad_(False)
    else:
        if hasattr(gaussians, "_xyz") and isinstance(gaussians._xyz, torch.nn.Parameter):
            gaussians._xyz.requires_grad_(bool(trainable))
        if hasattr(gaussians, "_rotation") and isinstance(gaussians._rotation, torch.nn.Parameter):
            gaussians._rotation.requires_grad_(bool(trainable))

    for name in ("_opacity", "_scaling", "_features_dc", "_features_rest", "_albedo", "_roughness", "_metallic"):
        p = getattr(gaussians, name, None)
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad_(bool(trainable))

def _generate_ground_pcd_from_plane(
    ground_plane_json: str,
    ground_texture_path: str | None,
    num_points: int,
    seed: int = 0,
    height_jitter: float = 0.0,
) -> "BasicPointCloud":
    """
    Generate a dense point cloud on the ground plane defined by ground_plane.json.

    The generated points can be appended to the object-only --gt_ply to create a single
    GaussianModel where ground is also PBR-shaded and has learnable materials.

    Notes:
    - Points are sampled uniformly in the plane UV bounds.
    - Normals are set to the plane normal.
    - Colors are optionally initialized from ground_texture.png (same convention as
      reconstruct_ground_plane_texture.py). If texture is missing, defaults to gray.
    """
    from scene.gaussian_model import BasicPointCloud

    with open(ground_plane_json, "r") as f:
        meta = json.load(f)

    plane_n = np.asarray(meta["plane"]["n"], dtype=np.float32)
    plane_n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    p0 = np.asarray(meta["basis"]["p0"], dtype=np.float32)
    basis_u = np.asarray(meta["basis"]["u"], dtype=np.float32)
    basis_v = np.asarray(meta["basis"]["v"], dtype=np.float32)

    uv_min = np.asarray(meta["uv_bounds"]["min"], dtype=np.float32)  # [2]
    uv_max = np.asarray(meta["uv_bounds"]["max"], dtype=np.float32)  # [2]

    rng = np.random.default_rng(int(seed))
    uu = rng.uniform(float(uv_min[0]), float(uv_max[0]), size=(int(num_points), 1)).astype(np.float32)
    vv = rng.uniform(float(uv_min[1]), float(uv_max[1]), size=(int(num_points), 1)).astype(np.float32)

    points = p0[None, :] + uu * basis_u[None, :] + vv * basis_v[None, :]
    if height_jitter and float(height_jitter) != 0.0:
        points = points + rng.normal(loc=0.0, scale=float(height_jitter), size=points.shape).astype(np.float32) * plane_n[None, :]

    normals = np.repeat(plane_n[None, :], repeats=points.shape[0], axis=0).astype(np.float32)

    colors = np.ones_like(points, dtype=np.float32) * 0.5
    if ground_texture_path is not None and os.path.exists(ground_texture_path):
        try:
            from PIL import Image

            img = np.asarray(Image.open(ground_texture_path).convert("RGB"), dtype=np.float32) / 255.0
            tex_h, tex_w = img.shape[0], img.shape[1]

            u01 = (uu.squeeze(1) - uv_min[0]) / max(float(uv_max[0] - uv_min[0]), 1e-8)
            v01 = (vv.squeeze(1) - uv_min[1]) / max(float(uv_max[1] - uv_min[1]), 1e-8)
            u01 = np.clip(u01, 0.0, 1.0)
            v01 = np.clip(v01, 0.0, 1.0)

            # v axis is flipped in the PNG (top = +v_max).
            x = u01 * (tex_w - 1)
            y = (1.0 - v01) * (tex_h - 1)
            x0 = np.floor(x).astype(np.int32)
            y0 = np.floor(y).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, tex_w - 1)
            y1 = np.clip(y0 + 1, 0, tex_h - 1)
            wx = (x - x0.astype(np.float32))[:, None]
            wy = (y - y0.astype(np.float32))[:, None]

            c00 = img[y0, x0]
            c10 = img[y0, x1]
            c01 = img[y1, x0]
            c11 = img[y1, x1]
            colors = (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11
            colors = np.clip(colors.astype(np.float32), 0.0, 1.0)
        except Exception as e:
            print(f"[Warning] Failed to sample ground texture for init colors: {e}. Using gray.")

    return BasicPointCloud(points=points, colors=colors, normals=normals)

class _GroundPlaneMeta:
    def __init__(
        self,
        plane_normal: torch.Tensor,
        plane_d: torch.Tensor,
        plane_origin: torch.Tensor,
        plane_u: torch.Tensor,
        plane_v: torch.Tensor,
        uv_min: torch.Tensor,
        uv_max: torch.Tensor,
    ):
        self.plane_normal = plane_normal
        self.plane_d = plane_d
        self.plane_origin = plane_origin
        self.plane_u = plane_u
        self.plane_v = plane_v
        self.uv_min = uv_min
        self.uv_max = uv_max

def _load_ground_plane_meta(json_path: str, device: str = "cuda") -> _GroundPlaneMeta:
    with open(json_path, "r") as f:
        meta = json.load(f)
    n = torch.tensor(meta["plane"]["n"], dtype=torch.float32, device=device)
    n = n / (torch.norm(n) + 1e-12)
    d = torch.tensor(float(meta["plane"]["d"]), dtype=torch.float32, device=device)
    p0 = torch.tensor(meta["basis"]["p0"], dtype=torch.float32, device=device)
    u = torch.tensor(meta["basis"]["u"], dtype=torch.float32, device=device)
    v = torch.tensor(meta["basis"]["v"], dtype=torch.float32, device=device)
    uv_min = torch.tensor(meta["uv_bounds"]["min"], dtype=torch.float32, device=device)
    uv_max = torch.tensor(meta["uv_bounds"]["max"], dtype=torch.float32, device=device)
    return _GroundPlaneMeta(
        plane_normal=n,
        plane_d=d,
        plane_origin=p0,
        plane_u=u,
        plane_v=v,
        uv_min=uv_min,
        uv_max=uv_max,
    )

def _ground_hit_mask_from_meta(
    meta: _GroundPlaneMeta,
    ray_dirs: torch.Tensor,      # [H,W,3]
    camera_center: torch.Tensor, # [3]
) -> torch.Tensor:
    """
    Compute a binary mask of pixels whose camera rays hit the ground plane and land inside UV bounds.
    Returns a float mask of shape [1,H,W].
    """
    # Plane equation: n·x + d = 0, ray: o + t*dir
    n = meta.plane_normal  # [3]
    o = camera_center.view(1, 1, 3)
    dir = ray_dirs

    n_dot_dir = (dir * n).sum(dim=-1)  # [H,W]
    parallel = torch.abs(n_dot_dir) < 1e-6
    n_dot_dir_safe = torch.where(parallel, torch.ones_like(n_dot_dir), n_dot_dir)
    n_dot_o = (o * n).sum(dim=-1) + meta.plane_d  # [1,1] broadcast
    t = -n_dot_o / n_dot_dir_safe  # [H,W]
    hit = (t > 0) & (~parallel)

    hit_points = o + t.unsqueeze(-1) * dir  # [H,W,3]
    rel = hit_points - meta.plane_origin.view(1, 1, 3)
    uu = (rel * meta.plane_u.view(1, 1, 3)).sum(dim=-1)
    vv = (rel * meta.plane_v.view(1, 1, 3)).sum(dim=-1)
    in_u = (uu >= meta.uv_min[0]) & (uu <= meta.uv_max[0])
    in_v = (vv >= meta.uv_min[1]) & (vv <= meta.uv_max[1])
    in_bounds = in_u & in_v

    out = (hit & in_bounds).to(torch.float32)  # [H,W]
    return out.unsqueeze(0)  # [1,H,W]

def _get_ray_dirs_world(viewpoint_cam) -> torch.Tensor:
    return compute_ray_directions_world_from_fov(
        image_height=viewpoint_cam.image_height,
        image_width=viewpoint_cam.image_width,
        fovx=viewpoint_cam.FoVx,
        fovy=viewpoint_cam.FoVy,
        world_view_transform=viewpoint_cam.world_view_transform,
        device="cuda",
    )

def _compute_background(
    ray_dirs: torch.Tensor,
    camera_center: torch.Tensor,
    env_light,
    ground_plane=None,
) -> torch.Tensor:
    """
    Compute background color for each pixel.

    If ground_plane is provided:
      - Rays hitting the ground plane → sample ground texture
      - Rays missing the ground → sample environment map (sky)
    Otherwise:
      - All rays sample environment map

    Args:
        ray_dirs: [H, W, 3] normalized ray directions (world space)
        camera_center: [3] camera position in world space
        env_light: EnvironmentLight module
        ground_plane: Optional GroundPlane module

    Returns:
        bg: [3, H, W] background color (ground + sky composite)
    """
    # Sample environment map for all rays (sky fallback)
    sky_color = env_light.sample(ray_dirs).permute(2, 0, 1)  # [3, H, W]

    if ground_plane is None:
        return sky_color

    # Sample ground plane
    ground_color, ground_mask = ground_plane.sample(ray_dirs, camera_center)
    ground_color = ground_color.permute(2, 0, 1)  # [3, H, W]
    ground_mask = ground_mask.unsqueeze(0).float()  # [1, H, W]

    # Composite: ground where hit, sky elsewhere
    bg = ground_color * ground_mask + sky_color * (1.0 - ground_mask)

    return bg

@torch.no_grad()
def _process_gt_mask(args, mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    Normalize/binarize/dilate the GT mask for consistent training semantics.
    Expects mask in [0,1] with shape [1,H,W].
    """
    if mask is None:
        return None

    out = mask
    if getattr(args, "mask_binarize", False):
        thr = float(getattr(args, "mask_threshold", 0.5))
        out = (out > thr).to(out.dtype)

    dilate_px = int(getattr(args, "mask_dilate_px", 0) or 0)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        out = torch.nn.functional.max_pool2d(
            out.unsqueeze(0), kernel_size=k, stride=1, padding=dilate_px
        ).squeeze(0)

    return torch.clamp(out, 0.0, 1.0)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def _safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:200] if len(s) > 200 else s

@torch.no_grad()
def _downsample_chw(x: torch.Tensor, max_hw: int = 256) -> torch.Tensor:
    """
    Downsample a [C,H,W] tensor to keep debugging dumps lightweight.
    """
    if x is None:
        return None
    if x.dim() != 3:
        return x
    c, h, w = x.shape
    if max(h, w) <= max_hw:
        return x
    scale = max_hw / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    y = torch.nn.functional.interpolate(
        x.unsqueeze(0),
        size=(nh, nw),
        mode="bilinear" if c != 1 else "bilinear",
        align_corners=False,
    ).squeeze(0)
    return y

@torch.no_grad()
def _maybe_dump_nonfinite(
    args,
    model_path: str,
    iteration: int,
    split_name: str,
    view_name: str,
    tensors: dict,
):
    """
    If enabled and any tensor contains NaN/Inf, write a debug dump to disk.
    """
    if not getattr(args, "debug_nonfinite_dump", False):
        return

    offending = {}
    for k, v in tensors.items():
        if v is None or not isinstance(v, torch.Tensor):
            continue
        if not torch.isfinite(v).all():
            offending[k] = v

    if not offending:
        return

    os.makedirs(os.path.join(model_path, "debug_nonfinite"), exist_ok=True)
    fname = (
        f"iter_{iteration:06d}_{_safe_slug(split_name)}_{_safe_slug(view_name)}.pt"
    )
    out_path = os.path.join(model_path, "debug_nonfinite", fname)

    payload = {
        "iteration": int(iteration),
        "split": split_name,
        "view_name": view_name,
        "tensor_keys": list(tensors.keys()),
        "offending_keys": list(offending.keys()),
        "stats": {},
    }

    for k, v in offending.items():
        finite = torch.isfinite(v)
        payload["stats"][k] = {
            "shape": tuple(v.shape),
            "dtype": str(v.dtype),
            "device": str(v.device),
            "finite_ratio": float(finite.float().mean().item()),
            "nan_ratio": float(torch.isnan(v).float().mean().item()),
            "posinf_ratio": float(torch.isposinf(v).float().mean().item()),
            "neginf_ratio": float(torch.isneginf(v).float().mean().item()),
        }

    if getattr(args, "debug_nonfinite_dump_full", False):
        # Full tensors can be huge; still move to CPU for portability.
        payload["tensors_full"] = {k: v.detach().cpu() for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    else:
        payload["tensors_downsampled"] = {
            k: _downsample_chw(v.detach()).cpu() if isinstance(v, torch.Tensor) and v.dim() == 3 else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in tensors.items()
        }

    torch.save(payload, out_path)
    print(f"[Debug] Non-finite detected; wrote dump: {out_path}")
    if getattr(args, "debug_nonfinite_raise", False):
        raise FloatingPointError(f"Non-finite values detected during eval; dump saved to {out_path}")

@torch.no_grad()
def _maybe_dump_env_map(args, model_path: str, iteration: int, env_light) -> None:
    if not getattr(args, "dump_env_map_on_eval", False):
        return
    out_dir = os.path.join(model_path, "debug_env_map")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"env_map_iter_{iteration:06d}.pt")
    payload = {
        "iteration": int(iteration),
        "env_map_res": int(getattr(env_light, "resolution", -1)),
        "env_map_shape": tuple(env_light.env_map.shape),
        "env_map": env_light.env_map.detach().cpu(),
        "solid_angle_weight": getattr(env_light, "solid_angle_weight", None).detach().cpu()
        if hasattr(env_light, "solid_angle_weight")
        else None,
    }
    torch.save(payload, out_path)
    env = env_light.env_map.detach()
    finite = torch.isfinite(env)
    finite_ratio = float(finite.float().mean().item())
    env_min = float(env[finite].min().item()) if finite_ratio > 0 else float("nan")
    env_max = float(env[finite].max().item()) if finite_ratio > 0 else float("nan")
    env_mean = float(env[finite].mean().item()) if finite_ratio > 0 else float("nan")
    print(
        f"[Debug] Saved env_map dump: {out_path} "
        f"(finite={finite_ratio:.6f}, min={env_min:.4f}, mean={env_mean:.4f}, max={env_max:.4f})"
    )

@torch.no_grad()
def _maybe_tb_log_env_map(args, tb_writer, iteration: int, env_light) -> None:
    if tb_writer is None or not getattr(args, "tb_log_env_map", False):
        return

    env = env_light.env_map.detach()  # [3, H, 2H]
    env = torch.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)

    # Raw visualization (clamped) + log visualization (more stable for HDR).
    env_vis = torch.clamp(env, 0.0, 1.0)
    env_log = torch.log1p(torch.clamp(env, min=0.0))
    env_log = env_log / (env_log.max() + 1e-8)

    tb_writer.add_image("debug/env_map_rgb", env_vis, iteration)
    tb_writer.add_image("debug/env_map_log_rgb", env_log, iteration)
    tb_writer.add_scalar("debug/env_map_mean", env.mean().item(), iteration)
    tb_writer.add_scalar("debug/env_map_max", env.max().item(), iteration)

@torch.no_grad()
def _run_pbr_eval(
    tb_writer,
    iteration: int,
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    dummy_color: torch.Tensor,
    env_light,
    ground_plane=None,
    log_gt: bool = False,
    args=None,
):
    print(f"\n[ITER {iteration}] Running Evaluation...")
    torch.cuda.empty_cache()

    validation_configs = (
        {'name': 'test', 'cameras': scene.getTestCameras()},
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 20, 5)]},
    )

    for config in validation_configs:
        cameras = config['cameras']
        if not cameras:
            continue

        l1_test = 0.0
        psnr_test = 0.0
        l1_test_masked = 0.0
        psnr_test_masked = 0.0
        masked_count = 0

        for cam_idx, viewpoint in enumerate(cameras):
            render_pkg = render(viewpoint, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)

            ray_dirs = _get_ray_dirs_world(viewpoint)
            bg_env = _compute_background(ray_dirs, viewpoint.camera_center, env_light, ground_plane)

            alpha_map = render_pkg["rend_alpha"]
            denom = alpha_map + 1e-6

            albedo = torch.clamp(render_pkg["gbuffer_albedo"] / denom, 0.0, 1.0)
            rough = torch.clamp(render_pkg["gbuffer_roughness"] / denom, 0.1, 0.999)
            metal = torch.clamp(render_pkg["gbuffer_metallic"] / denom, 0.0, 1.0)
            normal = render_pkg["rend_normal"] / denom
            depth_map = render_pkg.get("surf_depth")

            shaded = screen_space_pbr_shading(
                albedo, rough, metal,
                normal, depth_map,
                viewpoint.camera_center, viewpoint.world_view_transform,
                env_light=env_light,
                ray_dirs_world=ray_dirs,
            )

            pred = shaded * alpha_map + bg_env * (1.0 - alpha_map)
            pred = torch.clamp(pred, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            gt_mask = viewpoint.gt_alpha_mask.to("cuda") if viewpoint.gt_alpha_mask is not None else None

            if args is not None:
                _maybe_dump_nonfinite(
                    args=args,
                    model_path=scene.model_path,
                    iteration=iteration,
                    split_name=config["name"],
                    view_name=viewpoint.image_name,
                    tensors={
                        "pred": pred,
                        "gt_image": gt_image,
                        "alpha_map": alpha_map,
                        "bg_env": bg_env,
                        "shaded_obj": shaded,
                        "albedo": albedo,
                        "roughness": rough,
                        "metallic": metal,
                        "normal": normal,
                        "depth": depth_map,
                    },
                )

            l1_test += l1_loss(pred, gt_image).item()
            psnr_test += psnr(pred, gt_image).mean().item()
            if gt_mask is not None:
                l1_test_masked += l1_loss(pred, gt_image, mask=gt_mask).item()
                psnr_test_masked += psnr(pred, gt_image, mask=gt_mask).mean().item()
                masked_count += 1

            if tb_writer and (cam_idx < 4):
                prefix = f"{config['name']}_view_{viewpoint.image_name}"
                tb_writer.add_image(f"{prefix}/0_render_composite", pred, iteration)
                if log_gt:
                    tb_writer.add_image(f"{prefix}/0_gt", gt_image, iteration)
                tb_writer.add_image(f"{prefix}/1_albedo", albedo, iteration)
                tb_writer.add_image(f"{prefix}/2_roughness", rough.repeat(3, 1, 1), iteration)
                tb_writer.add_image(f"{prefix}/3_metallic", metal.repeat(3, 1, 1), iteration)
                tb_writer.add_image(f"{prefix}/4_pbr_shaded_obj", shaded, iteration)
                tb_writer.add_image(f"{prefix}/7_alpha", alpha_map.repeat(3, 1, 1), iteration)
                if viewpoint.gt_alpha_mask is not None:
                    tb_writer.add_image(
                        f"{prefix}/8_gt_alpha_mask",
                        viewpoint.gt_alpha_mask.to("cuda").repeat(3, 1, 1),
                        iteration,
                    )

        l1_test /= len(cameras)
        psnr_test /= len(cameras)
        msg = f"  [ITER {iteration}] {config['name']} PSNR: {psnr_test:.4f}"
        if masked_count > 0:
            l1_test_masked /= masked_count
            psnr_test_masked /= masked_count
            msg += f" | PSNR(mask): {psnr_test_masked:.4f}"
        print(msg)

        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)
            if masked_count > 0:
                tb_writer.add_scalar(f"{config['name']}/l1_loss_masked", l1_test_masked, iteration)
                tb_writer.add_scalar(f"{config['name']}/psnr_masked", psnr_test_masked, iteration)

    torch.cuda.empty_cache()

def training_pbr_static(dataset, opt, pipe, args):
    if args.gt_ply is None:
        raise ValueError("Error: --gt_ply argument is required for static geometry training.")

    tb_writer = prepare_output_and_logger(dataset)
    
    # 1. Initialize Gaussians from Dense PLY
    print(f"Loading dense GT point cloud from: {args.gt_ply}")
    pcd = fetchPly(args.gt_ply)

    # Optional: learn additional "unfixed" Gaussians for finite-depth background geometry (walls/room/etc.).
    # This replaces the legacy ground-plane-as-gaussians concept.
    unfixed_gaussians = bool(getattr(args, "unfixed_gaussians", False))
    if bool(getattr(args, "ground_as_gaussians", False)):
        print("[Deprecated] --ground_as_gaussians is deprecated; use --unfixed_gaussians instead.")
        unfixed_gaussians = True
    
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=True)
    gaussians.roughness_min = float(getattr(args, "roughness_min", 0.02))
    gaussians.roughness_max = float(getattr(args, "roughness_max", 0.999))
    
    # Note: We use a placeholder spatial_lr_scale=1.0 initially.
    # It will be updated after Scene creation when we know the true extent.
    gaussians.create_from_dense_pcd(pcd, spatial_lr_scale=1.0)

    # Optional: override initial roughness to a chosen physical value in [0, 1].
    # Internally, _roughness is passed through sigmoid and then clamped.
    roughness_init = getattr(args, "roughness_init", None)
    if roughness_init is not None:
        r = float(roughness_init)
        r = max(1e-6, min(1.0 - 1e-6, r))
        with torch.no_grad():
            gaussians._roughness.data.fill_(float(inverse_sigmoid(torch.tensor(r)).item()))
        print(f"Initialized roughness to {r} (pre-sigmoid={gaussians._roughness.data.mean().item():.4f})")
    
    # 2. Initialize Scene
    # Since gaussians._xyz is now populated, Scene will NOT re-initialize them from COLMAP.
    scene = Scene(dataset, gaussians)
    
    # Update spatial_lr_scale with the correct extent from Scene
    gaussians.spatial_lr_scale = scene.cameras_extent
    print(f"Updated spatial_lr_scale to {scene.cameras_extent}")

    # 3. Setup Optimizer (Fixed Geometry Mode)
    # This locks XYZ and Rotation, but allows Scale, Opacity, and PBR to be optimized.
    gaussians.training_setup_fixed_geometry_pbr_only(opt)

    # 3.5 Setup Unfixed (Background) Gaussians (optional, SH-only + densify)
    gaussians_unfixed = None
    if unfixed_gaussians:
        unfixed_num_points = int(getattr(args, "unfixed_num_points", 200_000) or 200_000)
        unfixed_seed = int(getattr(args, "unfixed_seed", 0) or 0)
        unfixed_exclude_margin_ratio = float(getattr(args, "unfixed_exclude_object_aabb_margin_ratio", 0.02) or 0.02)
        obj_pts = np.asarray(pcd.points)
        obj_aabb_min = obj_pts.min(axis=0)
        obj_aabb_max = obj_pts.max(axis=0)
        exclude_margin = float(scene.cameras_extent) * max(0.0, unfixed_exclude_margin_ratio)
        pcd_unfixed = _init_unfixed_pcd_from_dataset(
            source_path=dataset.source_path,
            num_points=unfixed_num_points,
            extent=float(scene.cameras_extent),
            seed=unfixed_seed,
            exclude_aabb_min=obj_aabb_min,
            exclude_aabb_max=obj_aabb_max,
            exclude_margin=exclude_margin,
        )
        gaussians_unfixed = GaussianModel(dataset.sh_degree, use_pbr=False)
        gaussians_unfixed.create_from_pcd(pcd_unfixed, spatial_lr_scale=scene.cameras_extent)
        gaussians_unfixed.training_setup(opt)
        print(f"[Unfixed] Enabled: N={gaussians_unfixed.get_xyz.shape[0]} (SH-only, densify enabled)")

    # PBR-only training does not supervise SH color output; keep background black so
    # G-buffer maps are clean premultiplied attributes (no background offset).
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    dummy_color = torch.zeros((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")

    # 4. Environment Light
    env_light = EnvironmentLight(args.env_map, resolution=args.env_map_res).cuda()
    if getattr(args, "env_light_pth", None):
        ckpt_path = str(args.env_light_pth)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--env_light_pth not found: {ckpt_path}")
        env_light.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        env_light = env_light.cuda()
        print(f"Loaded env_light checkpoint: {ckpt_path}")

    freeze_env_light = bool(getattr(args, "freeze_env_light", False))
    if freeze_env_light:
        for p in env_light.parameters():
            p.requires_grad_(False)
        env_light_optimizer = None
        print("Environment light is frozen (no optimization).")
    else:
        env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=opt.env_light_lr)
        if not args.no_env_gradient_scaling:
            env_light.register_gradient_scaling_hook()

    # EnvMap warmup: freeze ALL Gaussians so only env_map is updated.
    # We do this for BOTH object Gaussians and unfixed/background Gaussians (if enabled).
    if (not freeze_env_light) and int(getattr(args, "env_warmup_iters", 0) or 0) > 0:
        _set_gaussians_trainable(gaussians, False, freeze_geometry=True)
        if gaussians_unfixed is not None:
            _set_gaussians_trainable(gaussians_unfixed, False, freeze_geometry=False)
        print(f"[Warmup] Freezing ALL Gaussians for the first {int(args.env_warmup_iters)} iterations (EnvMap-only).")

    # Optional: anchor env_map to the pretrained initialization to reduce lighting/material ambiguity.
    env_map_ref = None
    env_prior_weight = float(getattr(args, "env_light_prior_weight", 0.0) or 0.0)
    if (not freeze_env_light) and (env_prior_weight > 0) and getattr(args, "env_light_pth", None):
        with torch.no_grad():
            env_map_ref = torch.nan_to_num(env_light.env_map.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Enabled env_light prior: weight={env_prior_weight}")

    scene_extent = float(scene.cameras_extent)

    # 4.5 Ground Plane (optional, for finite-depth backgrounds like checkerboard floor).
    # Note: If unfixed_gaussians is enabled, the unfixed Gaussians should explain finite-depth backgrounds instead.
    ground_plane = None
    if getattr(args, "ground_plane_json", None) and (not unfixed_gaussians):
        json_path = str(args.ground_plane_json)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"--ground_plane_json not found: {json_path}")

        # Determine texture path
        tex_path = getattr(args, "ground_texture", None)
        if tex_path is None:
            # Default: look for ground_texture.png in the same directory
            tex_path = os.path.join(os.path.dirname(json_path), "ground_texture.png")
        if not os.path.exists(tex_path):
            raise FileNotFoundError(f"Ground texture not found: {tex_path}")

        ground_plane = GroundPlane(json_path=json_path, texture_path=tex_path).cuda()
        print(f"[GroundPlane] Initialized with plane from {json_path}")

    ground_meta = None

    # 5. Training Loop
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_total_for_log = 0.0
    ema_recon_for_log = 0.0
    ema_env_for_log = 0.0
    ema_reg_for_log = 0.0
    
    # Early Stopping State
    best_window_loss = float('inf')
    patience_counter = 0
    recent_losses = []
    check_interval = args.early_stopping_interval  # Check every N steps

    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Training Static PBR")

    first_eval_iter = min(args.test_iterations) if getattr(args, "test_iterations", None) else None

    if getattr(args, "eval_first", False):
        _maybe_dump_env_map(args=args, model_path=scene.model_path, iteration=0, env_light=env_light)
        _maybe_tb_log_env_map(args=args, tb_writer=tb_writer, iteration=0, env_light=env_light)
        _run_pbr_eval(
            tb_writer=tb_writer,
            iteration=0,
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            dummy_color=dummy_color,
            env_light=env_light,
            ground_plane=ground_plane,
            log_gt=True,
            args=args,
        )

    first_iter = 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Transition out of warmup: unfreeze Gaussians.
        warmup_iters = int(getattr(args, "env_warmup_iters", 0) or 0)
        if (not freeze_env_light) and warmup_iters > 0 and (iteration == warmup_iters + 1):
            _set_gaussians_trainable(gaussians, True, freeze_geometry=True)
            # After warmup, allow the object Gaussians to optimize SH/color too (even if not supervised by default).
            gaussians.training_setup_fixed_geometry(opt)
            if gaussians_unfixed is not None:
                _set_gaussians_trainable(gaussians_unfixed, True, freeze_geometry=False)
                gaussians_unfixed.training_setup(opt)
            print(f"[Warmup] Unfroze Gaussians at iter={iteration}; now optimizing Gaussians + EnvMap.")

        # Update learning rate (mainly for Opacity/SH, since XYZ is locked)
        gaussians.update_learning_rate(iteration)

        if gaussians_unfixed is not None:
            gaussians_unfixed.update_learning_rate(iteration)

        # EnvMap Warmup Logic (disabled when env_light is frozen)
        is_env_warmup = (not freeze_env_light) and (iteration <= args.env_warmup_iters)

        batch_cams = int(getattr(args, "batch_cams", 1) or 1)
        batch_cams = max(1, batch_cams)

        # Global regularizers (apply once per iteration, not per camera).
        if freeze_env_light:
            env_tv_loss = torch.tensor(0.0, device="cuda")
            env_smooth_loss = torch.tensor(0.0, device="cuda")
        else:
            env_tv_loss = opt.lambda_env_tv * env_light.tv_loss_weighted()
            env_smooth_loss = getattr(args, "lambda_env_smooth", 0.0) * env_light.smoothness_loss_weighted()

        env_prior_loss = torch.tensor(0.0, device="cuda")
        if (not freeze_env_light) and (env_map_ref is not None) and (env_prior_weight > 0) and (not is_env_warmup):
            w = env_light.solid_angle_weight  # [1,H,W]
            diff = env_light.env_map - env_map_ref
            if getattr(args, "env_light_prior_log_space", False):
                a = torch.log1p(torch.clamp(env_light.env_map, min=0.0))
                b = torch.log1p(torch.clamp(env_map_ref, min=0.0))
                diff = a - b
            env_prior_loss = env_prior_weight * ((diff * diff) * w).sum() / (w.sum() * 3.0 + 1e-8)

        scale_reg_loss = torch.tensor(0.0, device="cuda")
        lambda_scale_reg = float(getattr(args, "lambda_scale_reg", 0.0) or 0.0)
        if lambda_scale_reg > 0 and (not is_env_warmup):
            log_scale_max = gaussians._scaling.max(dim=1).values
            scale_thresh = float(getattr(args, "scale_reg_max_ratio", 0.1)) * scene_extent
            log_thresh = float(np.log(max(scale_thresh, 1e-12)))
            log_scale_max = torch.nan_to_num(log_scale_max, nan=log_thresh, posinf=log_thresh + 10.0, neginf=-20.0)
            scale_over_log = torch.relu(log_scale_max - log_thresh)
            scale_reg_loss = lambda_scale_reg * (scale_over_log ** 2).mean()

        # Gradient accumulation over multiple cameras reduces variance for non-Lambertian cues.
        gaussians.optimizer.zero_grad(set_to_none=True)
        if env_light_optimizer is not None:
            env_light_optimizer.zero_grad(set_to_none=True)

        recon_loss_sum = 0.0
        l1_sum = 0.0
        ssim_term_sum = 0.0
        alpha_sup_sum = 0.0
        pbr_reg_sum = 0.0
        pbr_reg_unscaled_sum = 0.0
        obj_cov_sum = 0.0
        alpha_mean_sum = 0.0
        weight_mean_sum = 0.0

        # Keep last-camera tensors for logging images/diagnostics.
        viewpoint_cam = None
        alpha_map = None
        obj_mask = None
        recon_weight = None
        pbr_losses = {"total_pbr_reg": torch.tensor(0.0, device="cuda")}
        pbr_reg_loss = torch.tensor(0.0, device="cuda")
        alpha_sup_loss = torch.tensor(0.0, device="cuda")
        gbuffer_albedo = None
        gbuffer_roughness = None
        gbuffer_metallic = None
        unfixed_stats = []
        unfixed_render_pkg = None

        warned_gt_mask_comp = False
        for _ in range(batch_cams):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)
            alpha_map = render_pkg.get("rend_alpha")
            if alpha_map is None:
                raise RuntimeError("render_pkg missing 'rend_alpha'")

            mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else None
            mask = _process_gt_mask(args, mask)

            ray_dirs = _get_ray_dirs_world(viewpoint_cam)
            sky = env_light.sample(ray_dirs).permute(2, 0, 1)  # [3,H,W]
            if gaussians_unfixed is not None:
                unfixed_render_pkg = render(viewpoint_cam, gaussians_unfixed, pipe, background, render_pbr=False)
                alpha_bg = unfixed_render_pkg.get("rend_alpha")
                if alpha_bg is None:
                    raise RuntimeError("unfixed render_pkg missing 'rend_alpha'")
                bg_render = unfixed_render_pkg["render"]
                bg_env = bg_render + sky * (1.0 - alpha_bg)
                unfixed_stats.append(
                    (
                        unfixed_render_pkg["viewspace_points"],
                        unfixed_render_pkg["visibility_filter"],
                        unfixed_render_pkg["radii"],
                    )
                )
            else:
                bg_env = _compute_background(ray_dirs, viewpoint_cam.camera_center, env_light, ground_plane)
            gt_image = viewpoint_cam.original_image.cuda()

            gbuffer_albedo_pm = render_pkg.get("gbuffer_albedo")
            gbuffer_roughness_pm = render_pkg.get("gbuffer_roughness")
            gbuffer_metallic_pm = render_pkg.get("gbuffer_metallic")
            gbuffer_normal_pm = render_pkg.get("rend_normal")
            gbuffer_depth = render_pkg.get("surf_depth")
            if gbuffer_albedo_pm is None:
                raise RuntimeError("render_pbr=True but missing G-buffer outputs")

            eps = 1e-6
            denom = alpha_map + eps
            gbuffer_albedo = torch.clamp(gbuffer_albedo_pm / denom, 0.0, 1.0)
            gbuffer_roughness = torch.clamp(gbuffer_roughness_pm / denom, 0.1, 0.999)
            gbuffer_metallic = torch.clamp(gbuffer_metallic_pm / denom, 0.0, 1.0)
            gbuffer_normal = gbuffer_normal_pm / denom

            shaded_obj = screen_space_pbr_shading(
                gbuffer_albedo,
                gbuffer_roughness,
                gbuffer_metallic,
                gbuffer_normal,
                gbuffer_depth,
                viewpoint_cam.camera_center,
                viewpoint_cam.world_view_transform,
                env_light=env_light,
                ray_dirs_world=ray_dirs,
            )

            # Composite for reconstruction.
            # By default, composite with rendered alpha. If requested and GT mask exists, composite with masks:
            # - object: gt_alpha_mask
            alpha_for_comp = alpha_map
            if getattr(args, "composite_use_gt_mask", False) and (mask is not None):
                alpha_for_comp = mask

            pred = shaded_obj * alpha_for_comp + bg_env * (1.0 - alpha_for_comp)

            alpha_sup_loss = torch.tensor(0.0, device="cuda")
            if mask is not None:
                lambda_alpha = float(getattr(args, "lambda_alpha", 0.0) or 0.0)
                if lambda_alpha > 0 and (not is_env_warmup):
                    alpha_sup_loss = lambda_alpha * torch.abs(alpha_map - mask).mean()

            obj_mask = mask if mask is not None else alpha_map.detach()
            if is_env_warmup:
                recon_weight = torch.ones_like(alpha_map)
            elif mask is not None:
                if getattr(args, "lambda_bg", None) is None:
                    # If unfixed background Gaussians are enabled, we typically want to supervise background too.
                    if gaussians_unfixed is not None:
                        bg_w = 1.0
                    else:
                        bg_w = 1.0 if getattr(args, "supervise_background", False) else 0.0
                else:
                    bg_w = float(args.lambda_bg)
                bg_w = max(0.0, bg_w)
                recon_weight = mask + bg_w * (1.0 - mask)
                if getattr(opt, "lambda_pbr", 0.0) > 0:
                    recon_weight = recon_weight + opt.lambda_pbr * obj_mask
            else:
                recon_weight = torch.ones_like(alpha_map)
                if getattr(opt, "lambda_pbr", 0.0) > 0:
                    recon_weight = recon_weight + opt.lambda_pbr * obj_mask

            reg_mask = mask if mask is not None else alpha_map.detach()
            if is_env_warmup:
                pbr_losses = {"total_pbr_reg": torch.tensor(0.0, device="cuda")}
                pbr_reg_loss = torch.tensor(0.0, device="cuda")
            else:
                pbr_losses = compute_pbr_losses(
                    gbuffer_albedo,
                    gbuffer_roughness,
                    gbuffer_metallic,
                    alpha_map=reg_mask,
                    lambda_albedo_smooth=args.lambda_albedo_smooth,
                    lambda_roughness_smooth=args.lambda_roughness_smooth,
                    lambda_metallic_smooth=args.lambda_metallic_smooth,
                    lambda_metallic_prior=args.lambda_metallic_prior,
                    lambda_roughness_prior=args.lambda_roughness_prior,
                    lambda_albedo_chroma=args.lambda_albedo_chroma,
                )
                pbr_reg_loss = opt.lambda_pbr_reg * pbr_losses["total_pbr_reg"]

            Ll1 = l1_loss(pred, gt_image, mask=recon_weight)
            ssim_val = ssim(pred.unsqueeze(0), gt_image.unsqueeze(0), mask=recon_weight.unsqueeze(0))
            recon_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

            cam_loss = opt.lambda_rgb * recon_loss + alpha_sup_loss + pbr_reg_loss
            if not torch.isfinite(cam_loss):
                raise FloatingPointError("Non-finite cam_loss detected during minibatch accumulation.")
            (cam_loss / float(batch_cams)).backward()

            recon_loss_sum += float(recon_loss.detach().cpu().item())
            l1_sum += float(Ll1.detach().cpu().item())
            ssim_term_sum += float((1.0 - ssim_val.detach()).cpu().item())
            alpha_sup_sum += float(alpha_sup_loss.detach().cpu().item())
            pbr_reg_sum += float(pbr_reg_loss.detach().cpu().item())
            pbr_reg_unscaled_sum += float(pbr_losses["total_pbr_reg"].detach().cpu().item())
            obj_cov_sum += float((obj_mask > 0.5).float().mean().detach().cpu().item())
            alpha_mean_sum += float(alpha_map.mean().detach().cpu().item())
            weight_mean_sum += float(recon_weight.mean().detach().cpu().item())

        global_loss = env_tv_loss + env_smooth_loss + env_prior_loss + scale_reg_loss
        if not torch.isfinite(global_loss):
            raise FloatingPointError("Non-finite global regularizer loss detected.")
        global_loss.backward()

        recon_loss = torch.tensor(recon_loss_sum / float(batch_cams), device="cuda")
        Ll1 = torch.tensor(l1_sum / float(batch_cams), device="cuda")
        ssim_val = torch.tensor(1.0 - (ssim_term_sum / float(batch_cams)), device="cuda")
        alpha_sup_loss = torch.tensor(alpha_sup_sum / float(batch_cams), device="cuda")
        pbr_reg_loss = torch.tensor(pbr_reg_sum / float(batch_cams), device="cuda")

        total_loss = (
            opt.lambda_rgb * recon_loss
            + env_tv_loss
            + env_smooth_loss
            + env_prior_loss
            + scale_reg_loss
            + alpha_sup_loss
            + pbr_reg_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite total_loss detected after minibatch accumulation.")

        iter_end.record()

        # Optimize
        with torch.no_grad():
            # Logging
            ema_total_for_log = 0.4 * total_loss.item() + 0.6 * ema_total_for_log
            ema_recon_for_log = 0.4 * recon_loss.item() + 0.6 * ema_recon_for_log
            ema_env_for_log = 0.4 * env_tv_loss.item() + 0.6 * ema_env_for_log
            ema_reg_for_log = 0.4 * pbr_reg_loss.item() + 0.6 * ema_reg_for_log
            
            if iteration % 10 == 0:
                bg_pts = int(gaussians_unfixed.get_xyz.shape[0]) if gaussians_unfixed is not None else 0
                progress_bar.set_postfix({
                    "Status": "Warmup" if is_env_warmup else "Normal",
                    "Tot": f"{ema_total_for_log:.{5}f}",
                    "Recon": f"{ema_recon_for_log:.{5}f}",
                    "Env": f"{ema_env_for_log:.{2}e}",
                    "Reg": f"{ema_reg_for_log:.{2}e}",
                    "ObjPts": f"{len(gaussians.get_xyz)}",
                    "BgPts": f"{bg_pts}",
                })
                progress_bar.update(10)

            if args.log_interval > 0 and (iteration % args.log_interval == 0):
                env_tv_unscaled = env_light.tv_loss_weighted().item()
                pbr_reg_unscaled = pbr_reg_unscaled_sum / float(batch_cams)
                obj_cov = obj_cov_sum / float(batch_cams)
                scale_max_all = gaussians.get_scaling.max(dim=1).values
                scale_max_val = torch.nan_to_num(scale_max_all, nan=0.0, posinf=1e9, neginf=0.0).max().item()
                env_mean = env_light.env_map.mean().item()
                env_max = env_light.env_map.max().item()
                alpha_mean = alpha_mean_sum / float(batch_cams)
                w_mean = weight_mean_sum / float(batch_cams)
                print(
                    f"\n[ITER {iteration}] total={total_loss.item():.6f} "
                    f"(lambda_rgb*recon={opt.lambda_rgb * recon_loss.item():.6f}, "
                    f"lambda_env_tv*tv={env_tv_loss.item():.6f}, "
                    f"lambda_env_smooth*smooth={env_smooth_loss.item():.6f}, "
                    f"lambda_scale_reg*scale={scale_reg_loss.item():.6f}, "
                    f"lambda_alpha*alpha={alpha_sup_loss.item():.6f}, "
                    f"lambda_pbr_reg*reg={pbr_reg_loss.item():.6f}) | "
                    f"recon={recon_loss.item():.6f} (L1={Ll1.item():.6f}, 1-SSIM={(1.0-ssim_val).item():.6f}) | "
                    f"tv_unscaled={env_tv_unscaled:.6e} reg_unscaled={pbr_reg_unscaled:.6e} | "
                    f"env_mean={env_mean:.3f} env_max={env_max:.3f} | "
                    f"scale_max={scale_max_val:.3f} | "
                    f"obj_cov={obj_cov:.3f} alpha_mean={alpha_mean:.3f} w_mean={w_mean:.3f}"
                )

            # Early Stopping Check (Relative Percentage Strategy)
            if args.enable_early_stopping:
                recent_losses.append(total_loss.item())
                if iteration % check_interval == 0:
                    avg_window_loss = sum(recent_losses) / len(recent_losses)
                    recent_losses = [] # Reset buffer

                    if best_window_loss == float('inf'):
                         best_window_loss = avg_window_loss
                    else:
                        # Calculate relative improvement: (Old - New) / Old
                        # Example: 0.01 means 1% improvement
                        rel_improvement = (best_window_loss - avg_window_loss) / best_window_loss
                        
                        if rel_improvement > args.early_stopping_min_delta:
                            # Significant improvement found
                            best_window_loss = avg_window_loss
                            patience_counter = 0 # Reset patience
                        else:
                            # Improvement too small or negative
                            patience_counter += 1
                            if patience_counter >= args.early_stopping_patience:
                                print(f"\n[Early Stopping] No relative improvement > {args.early_stopping_min_delta:.2%} for {patience_counter * check_interval} steps.")
                                print(f"Best Window Loss: {best_window_loss:.5f}, Current: {avg_window_loss:.5f}")
                                print("Stopping training and saving final model...")
                                
                                scene.save(iteration)
                                torch.save(env_light.state_dict(), os.path.join(scene.model_path, f"env_light_{iteration}.pth"))
                                progress_bar.close()
                                print("Training complete (Early Stopped).")
                                return # Exit function directly

            if tb_writer:
                tb_writer.add_scalar('train/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('train/recon_loss', recon_loss.item(), iteration)
                tb_writer.add_scalar('train/recon_l1', Ll1.item(), iteration)
                tb_writer.add_scalar('train/recon_ssim_term', (1.0 - ssim_val).item(), iteration)
                tb_writer.add_scalar('train/env_tv_loss', env_tv_loss.item(), iteration)
                tb_writer.add_scalar('train/alpha_sup_loss', alpha_sup_loss.item(), iteration)
                tb_writer.add_scalar('train/pbr_reg_loss', pbr_reg_loss.item(), iteration)

                # Unscaled components (useful for tuning lambdas)
                tb_writer.add_scalar('train_unscaled/env_tv', env_light.tv_loss_weighted().item(), iteration)
                tb_writer.add_scalar('train_unscaled/pbr_reg', pbr_losses["total_pbr_reg"].item(), iteration)

                for k, v in pbr_losses.items():
                    if k == "total_pbr_reg":
                        continue
                    tb_writer.add_scalar(f"train_unscaled/pbr_reg/{k}", v.item(), iteration)

                # Weight/mask diagnostics
                obj_coverage = (obj_mask > 0.5).float().mean().item()
                alpha_mean = alpha_map.mean().item()
                w_mean = recon_weight.mean().item()
                tb_writer.add_scalar("train_diag/obj_coverage", obj_coverage, iteration)
                tb_writer.add_scalar("train_diag/alpha_mean", alpha_mean, iteration)
                tb_writer.add_scalar("train_diag/recon_weight_mean", w_mean, iteration)

                # Material stats (object region only, if any)
                if obj_coverage > 0:
                    m = (obj_mask > 0.5).expand_as(gbuffer_albedo)
                    tb_writer.add_scalar("train_diag/albedo_mean_obj", gbuffer_albedo[m].mean().item(), iteration)
                    tb_writer.add_scalar("train_diag/roughness_mean_obj", gbuffer_roughness[obj_mask > 0.5].mean().item(), iteration)
                    tb_writer.add_scalar("train_diag/metallic_mean_obj", gbuffer_metallic[obj_mask > 0.5].mean().item(), iteration)
                tb_writer.add_scalar('train_loss_patches/recon_loss', recon_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/env_tv_loss', env_tv_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/env_prior_loss', env_prior_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/pbr_reg_pre_scale', pbr_losses["total_pbr_reg"].item(), iteration)
                tb_writer.add_scalar('train_stats/alpha_mean', alpha_map.mean().item(), iteration)
                tb_writer.add_scalar('train_stats/obj_weight_mean', recon_weight.mean().item(), iteration)
                tb_writer.add_scalar('train_stats/albedo_mean', gbuffer_albedo.mean().item(), iteration)
                tb_writer.add_scalar('train_stats/roughness_mean', gbuffer_roughness.mean().item(), iteration)
                tb_writer.add_scalar('train_stats/metallic_mean', gbuffer_metallic.mean().item(), iteration)

                for k, v in pbr_losses.items():
                    if k == "total_pbr_reg":
                        continue
                    tb_writer.add_scalar(f"train_loss_patches/pbr_reg_terms/{k}", v.item(), iteration)

            # Save
            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Model...")
                scene.save(iteration)
                if gaussians_unfixed is not None:
                    unfixed_dir = os.path.join(scene.model_path, "unfixed_point_cloud", f"iteration_{iteration}")
                    os.makedirs(unfixed_dir, exist_ok=True)
                    gaussians_unfixed.save_ply(os.path.join(unfixed_dir, "point_cloud.ply"))
                if not freeze_env_light:
                    torch.save(env_light.state_dict(), os.path.join(scene.model_path, f"env_light_{iteration}.pth"))

            # Densification (unfixed/background only)
            if (gaussians_unfixed is not None) and (not is_env_warmup) and (iteration < opt.densify_until_iter):
                for viewspace_point_tensor, visibility_filter, radii in unfixed_stats:
                    gaussians_unfixed.max_radii2D[visibility_filter] = torch.max(
                        gaussians_unfixed.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians_unfixed.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration > opt.densify_from_iter) and (iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_unfixed.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if (iteration % opt.opacity_reset_interval == 0) or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians_unfixed.reset_opacity()

            # Step
            # [EnvMap Warmup] Freeze ALL Gaussians during warmup (EnvMap-only).
            # After warmup, train Gaussians + EnvMap together.
            if not is_env_warmup:
                gaussians.optimizer.step()
                if gaussians_unfixed is not None:
                    gaussians_unfixed.optimizer.step()
            
            # ALWAYS zero grad for Gaussians to prevent accumulation during warmup
            gaussians.optimizer.zero_grad(set_to_none=True)
            if gaussians_unfixed is not None:
                gaussians_unfixed.optimizer.zero_grad(set_to_none=True)
            
            # Step EnvMap (unless frozen), optionally only after a certain iteration / periodically.
            if not freeze_env_light:
                env_update_after = int(getattr(args, "env_update_after", 0) or 0)
                env_update_interval = int(getattr(args, "env_update_interval", 1) or 1)
                env_update_interval = max(1, env_update_interval)
                # Warmup is explicitly designed to update env_light every iteration.
                if is_env_warmup:
                    do_env_step = True
                else:
                    do_env_step = True
                    if env_update_after > 0 and iteration < env_update_after:
                        do_env_step = False
                    if env_update_interval > 1 and (iteration % env_update_interval != 0):
                        do_env_step = False
                if do_env_step:
                    env_light_optimizer.step()
                env_light_optimizer.zero_grad(set_to_none=True)

            env_clamp_min = getattr(args, "env_clamp_min", None)
            env_clamp_max = getattr(args, "env_clamp_max", None)
            if (not freeze_env_light) and ((env_clamp_min is not None) or (env_clamp_max is not None)):
                with torch.no_grad():
                    min_v = float(env_clamp_min) if env_clamp_min is not None else -float("inf")
                    max_v = float(env_clamp_max) if env_clamp_max is not None else float("inf")
                    env_light.env_map.data.clamp_(min=min_v, max=max_v)

            scale_clamp_ratio = getattr(args, "scale_clamp_max_ratio", None)
            if scale_clamp_ratio is not None:
                max_scale = float(scale_clamp_ratio) * scene_extent
                if max_scale > 0:
                    with torch.no_grad():
                        gaussians._scaling.data.clamp_(max=float(torch.log(torch.tensor(max_scale)).item()))

            # --- Evaluation and Image Logging ---
            if iteration in args.test_iterations:
                _maybe_dump_env_map(args=args, model_path=scene.model_path, iteration=iteration, env_light=env_light)
                _maybe_tb_log_env_map(args=args, tb_writer=tb_writer, iteration=iteration, env_light=env_light)
                _run_pbr_eval(
                    tb_writer=tb_writer,
                    iteration=iteration,
                    scene=scene,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    dummy_color=dummy_color,
                    env_light=env_light,
                    ground_plane=ground_plane,
                    log_gt=(first_eval_iter is not None and iteration == first_eval_iter),
                    args=args,
                )

            # NO Densification!
            
    progress_bar.close()
    print("Training complete.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Static PBR Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--gt_ply", type=str, required=True, help="Path to dense GT .ply file")
    parser.add_argument("--env_map", type=str, default=None, help="Initial HDR environment map")
    parser.add_argument(
        "--env_light_pth",
        type=str,
        default=None,
        help="Optional pretrained EnvironmentLight state_dict (.pth) to load before training (e.g., output of train_env_light.py).",
    )
    parser.add_argument(
        "--freeze_env_light",
        action="store_true",
        help="If set, do not optimize env_light (treat as fixed input); env warmup/regularization/checkpoint saving are disabled.",
    )
    parser.add_argument(
        "--env_light_prior_weight",
        type=float,
        default=0.0,
        help="If >0 and --env_light_pth is provided, add an L2 prior anchoring env_map to the pretrained initialization (reduces lighting/material ambiguity).",
    )
    parser.add_argument(
        "--env_light_prior_log_space",
        action="store_true",
        help="Compute env_light prior in log1p space (more stable when env_map is HDR-like).",
    )
    parser.add_argument(
        "--env_update_after",
        type=int,
        default=0,
        help="If >0, only start updating env_light at this iteration (still uses it for rendering before).",
    )
    parser.add_argument(
        "--env_update_interval",
        type=int,
        default=1,
        help="Update env_light every N iterations after env_update_after (warmup always updates).",
    )
    parser.add_argument(
        "--batch_cams",
        type=int,
        default=1,
        help="Number of random training cameras per iteration (gradient accumulation); improves stability for non-Lambertian cues.",
    )
    
    # PBR params
    parser.add_argument(
        "--lambda_rgb",
        type=float,
        default=1.0,
        help="Weight for full-image composite reconstruction loss (PBR object + skybox)",
    )
    parser.add_argument(
        "--lambda_pbr",
        type=float,
        default=0.1,
        help="Extra reconstruction weight on the object region (mask if available, else alpha)",
    )
    parser.add_argument("--lambda_pbr_reg", type=float, default=0.01)
    parser.add_argument("--env_light_lr", type=float, default=0.01)
    parser.add_argument("--lambda_env_tv", type=float, default=0.001)
    parser.add_argument(
        "--lambda_env_smooth",
        type=float,
        default=0.0,
        help="Additional L2 Laplacian smoothness regularizer for env_map (helps suppress speckle hotspots).",
    )
    parser.add_argument(
        "--env_clamp_min",
        type=float,
        default=None,
        help="Optional clamp min for env_map values after each optimizer step (e.g., 0.0).",
    )
    parser.add_argument(
        "--env_clamp_max",
        type=float,
        default=None,
        help="Optional clamp max for env_map values after each optimizer step (e.g., 5.0).",
    )
    parser.add_argument(
        "--lambda_alpha",
        type=float,
        default=0.0,
        help="Supervise rendered alpha (rend_alpha) to match gt_alpha_mask to prevent opacity cheating.",
    )
    parser.add_argument(
        "--lambda_scale_reg",
        type=float,
        default=0.0,
        help="Penalize oversized Gaussians to prevent scale blow-up; weight for (relu(scale_max - thresh)^2).",
    )
    parser.add_argument(
        "--scale_reg_max_ratio",
        type=float,
        default=0.1,
        help="Scale threshold as a ratio of scene extent for scale regularization (thresh = ratio * cameras_extent).",
    )
    parser.add_argument(
        "--scale_clamp_max_ratio",
        type=float,
        default=None,
        help="Optional hard clamp on Gaussian scales (ratio * cameras_extent), applied after each optimizer step.",
    )
    parser.add_argument("--no_env_gradient_scaling", action="store_true")
    parser.add_argument(
        "--roughness_min",
        type=float,
        default=0.02,
        help="Minimum roughness clamp (lower allows sharper highlights; too low can introduce noise).",
    )
    parser.add_argument(
        "--roughness_max",
        type=float,
        default=0.999,
        help="Maximum roughness clamp.",
    )
    parser.add_argument(
        "--albedo_lr",
        type=float,
        default=5e-4,
        help="Learning rate for albedo (lower helps prevent baking specular into albedo).",
    )
    parser.add_argument(
        "--roughness_lr",
        type=float,
        default=5e-4,
        help="Learning rate for roughness (higher helps explain highlights via roughness instead of albedo).",
    )
    parser.add_argument(
        "--metallic_lr",
        type=float,
        default=2e-4,
        help="Learning rate for metallic.",
    )
    parser.add_argument(
        "--supervise_background",
        action="store_true",
        help="Supervise full composite (object + background). If unset and gt_alpha_mask exists, L1/SSIM is computed only on the mask region to avoid black-background supervision.",
    )
    parser.add_argument(
        "--lambda_bg",
        type=float,
        default=None,
        help="Optional background reconstruction weight when gt_alpha_mask exists (weight for (1-mask)). "
             "If unset, defaults to 1.0 when --supervise_background is set, otherwise 0.0.",
    )
    parser.add_argument(
        "--composite_use_gt_mask",
        action="store_true",
        help="If set and gt_alpha_mask exists, composite pred with gt_alpha_mask instead of rendered alpha "
             "for reconstruction/evaluation (decouples background supervision from opacity artifacts).",
    )

    # Material regularization term weights (inside compute_pbr_losses, before global lambda_pbr_reg)
    parser.add_argument("--lambda_albedo_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_roughness_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_metallic_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_metallic_prior", type=float, default=0.001)
    parser.add_argument("--lambda_roughness_prior", type=float, default=0.001)
    parser.add_argument("--lambda_albedo_chroma", type=float, default=0.001)

    # Early Stopping Params
    parser.add_argument("--enable_early_stopping", action="store_true", help="Enable automatic early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of checks with no improvement before stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum relative improvement to be considered significant")
    parser.add_argument("--early_stopping_interval", type=int, default=500, help="Interval (iterations) to check for improvement")
    parser.add_argument(
        "--env_warmup_iters",
        type=int,
        default=1000,
        help="Warmup iterations: freeze ALL Gaussians and optimize ONLY the environment map (env_map).",
    )
    parser.add_argument("--env_map_res", type=int, default=1024, help="Resolution of the environment map (height). Width will be 2x height.")

    # Ground Plane (for finite-depth backgrounds like checkerboard floor)
    parser.add_argument(
        "--ground_plane_json",
        type=str,
        default=None,
        help="Path to ground_plane.json (output of reconstruct_ground_plane_texture.py). "
             "If provided, ground texture will be used for background pixels that hit the plane.",
    )
    parser.add_argument(
        "--ground_texture",
        type=str,
        default=None,
        help="Path to ground_texture.png. If not specified, will look for ground_texture.png "
             "in the same directory as ground_plane_json.",
    )

    # Unfixed (learnable) background Gaussians (finite-depth geometry, e.g., walls/room).
    # These Gaussians are SH-only (use_pbr=False) and can be densified/pruned.
    parser.add_argument(
        "--unfixed_gaussians",
        action="store_true",
        help="If set, learn an additional unfixed/background Gaussian set (SH-only) to explain finite-depth background geometry.",
    )
    parser.add_argument(
        "--unfixed_num_points",
        type=int,
        default=200_000,
        help="Initial number of unfixed/background points. If dataset point cloud exists (points3d.ply / points3D.ply), it will be sampled; otherwise random initialization is used.",
    )
    parser.add_argument(
        "--unfixed_seed",
        type=int,
        default=0,
        help="RNG seed for unfixed/background initialization.",
    )
    parser.add_argument(
        "--unfixed_exclude_object_aabb_margin_ratio",
        type=float,
        default=0.02,
        help="When initializing unfixed/background points, exclude points that fall inside the object AABB expanded by (ratio * cameras_extent).",
    )
    parser.add_argument(
        "--ground_as_gaussians",
        action="store_true",
        help="[Deprecated] Use --unfixed_gaussians instead. This flag is kept for backward compatibility and will enable --unfixed_gaussians.",
    )
    parser.add_argument(
        "--ground_num_points",
        type=int,
        default=200_000,
        help="[Deprecated] (unused) kept for backward compatibility.",
    )
    parser.add_argument(
        "--ground_seed",
        type=int,
        default=0,
        help="[Deprecated] (unused) kept for backward compatibility.",
    )
    parser.add_argument(
        "--ground_height_jitter",
        type=float,
        default=0.0,
        help="[Deprecated] (unused) kept for backward compatibility.",
    )
    parser.add_argument(
        "--lambda_ground",
        type=float,
        default=1.0,
        help="[Deprecated] (unused) kept for backward compatibility.",
    )

    # Save/Test
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument(
        "--test_interval",
        type=int,
        default=0,
        help="If >0, run evaluation every N iterations (overrides --test_iterations).",
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Print a detailed console loss breakdown every N iterations (0 disables)",
    )
    parser.add_argument(
        "--roughness_init",
        type=float,
        default=None,
        help="Optional initial roughness value in [0,1] for all points (before training).",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        help="If set, run evaluation once at iteration 0 (before any optimizer steps).",
    )
    parser.add_argument(
        "--debug_nonfinite_dump",
        action="store_true",
        help="If set, dump a debug .pt when NaN/Inf appears during evaluation.",
    )
    parser.add_argument(
        "--debug_nonfinite_dump_full",
        action="store_true",
        help="If set, include full-resolution tensors in the non-finite debug dump (can be very large).",
    )
    parser.add_argument(
        "--debug_nonfinite_raise",
        action="store_true",
        help="If set, raise an exception after writing a non-finite debug dump.",
    )
    parser.add_argument(
        "--dump_env_map_on_eval",
        action="store_true",
        help="If set, save env_map tensors to <model_path>/debug_env_map at each evaluation iteration.",
    )
    parser.add_argument(
        "--tb_log_env_map",
        action="store_true",
        help="If set, log env_map images/scalars to TensorBoard at each evaluation iteration.",
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.test_interval and args.test_interval > 0:
        args.test_iterations = list(range(args.test_interval, args.iterations + 1, args.test_interval))
        if args.iterations not in args.test_iterations:
            args.test_iterations.append(args.iterations)
        args.test_iterations = sorted(set(args.test_iterations))
    
    # Transfer args to opt
    opt = op.extract(args)
    opt.lambda_rgb = args.lambda_rgb
    opt.env_light_lr = args.env_light_lr
    opt.lambda_pbr = args.lambda_pbr
    opt.lambda_pbr_reg = args.lambda_pbr_reg
    opt.lambda_env_tv = args.lambda_env_tv
    opt.albedo_lr = args.albedo_lr
    opt.roughness_lr = args.roughness_lr
    opt.metallic_lr = args.metallic_lr
    
    training_pbr_static(lp.extract(args), opt, pp.extract(args), args)
