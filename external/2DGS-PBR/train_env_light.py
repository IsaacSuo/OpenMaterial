#!/usr/bin/env python3
"""
Standalone environment light (env_map) pretraining from the OBJECT region only.

Why:
- Without a sky mask, supervising background pixels will "bake" finite-depth geometry
  (e.g., ground/buildings) into env_map, producing patchwork artifacts.
- This script instead uses the foreground/object mask (gt_alpha_mask == 1) and fixed
  geometry from --gt_ply to learn an env_map that explains object appearance.

Key design:
- Fixed geometry from dense point cloud (--gt_ply), no densification.
- PBR shading with EnvironmentLight, supervised only on object pixels.
- Two-phase schedule:
  - Stage 1: env-only (gaussians frozen), optional low-res env_map.
  - Stage 2: alternating env/material steps with strong material regularization.
- Highlight robustness: optionally down-weight (or drop) the top-brightness object pixels
  to reduce "bake highlights into albedo" failure modes early on.
- Gauge fixing: optional env_map mean-luminance normalization to avoid scale drift.

Outputs:
- <model_path>/env_light_final.pth  (env_light.state_dict())
"""

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state, inverse_sigmoid
from utils.loss_utils import l1_loss, ssim, compute_pbr_losses
from utils.pbr_utils import EnvironmentLight, screen_space_pbr_shading, compute_ray_directions_world_from_fov


def prepare_output(args):
    if not args.model_path:
        unique = str(uuid.uuid4())[:10]
        args.model_path = os.path.join("./output/", unique)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args_env_light"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    print(f"Output folder: {args.model_path}")


def _get_ray_dirs_world(viewpoint_cam) -> torch.Tensor:
    return compute_ray_directions_world_from_fov(
        image_height=viewpoint_cam.image_height,
        image_width=viewpoint_cam.image_width,
        fovx=viewpoint_cam.FoVx,
        fovy=viewpoint_cam.FoVy,
        world_view_transform=viewpoint_cam.world_view_transform,
        device="cuda",
    )


@torch.no_grad()
def _process_mask(mask: torch.Tensor | None, binarize: bool, threshold: float) -> torch.Tensor | None:
    if mask is None:
        return None
    out = mask
    if binarize:
        out = (out > float(threshold)).to(out.dtype)
    return torch.clamp(out, 0.0, 1.0)


@torch.no_grad()
def _highlight_weight_map(
    gt_rgb: torch.Tensor,  # [3,H,W] in [0,1]
    obj_mask: torch.Tensor,  # [1,H,W] in [0,1]
    q: float,
    drop: float,
) -> torch.Tensor:
    """
    Return a per-pixel weight map in [0,1], where the brightest object pixels are downweighted.
    drop=1.0 -> drop highlights entirely, drop=0.0 -> no effect.
    """
    if drop <= 0:
        return obj_mask
    lum = 0.2126 * gt_rgb[0:1] + 0.7152 * gt_rgb[1:2] + 0.0722 * gt_rgb[2:3]
    vals = lum[obj_mask > 0.5]
    if vals.numel() < 16:
        return obj_mask
    thr = torch.quantile(vals, q=float(q))
    hi = (lum >= thr).to(obj_mask.dtype)
    w = obj_mask * (1.0 - float(drop) * hi)
    return torch.clamp(w, 0.0, 1.0)


@torch.no_grad()
def _gauge_normalize_env_mean_luminance(env_light: EnvironmentLight, target_mean: float) -> None:
    """
    Fix the global scale ambiguity by enforcing a target mean luminance (solid-angle weighted).
    """
    if target_mean is None:
        return
    target_mean = float(target_mean)
    if target_mean <= 0:
        return
    env = env_light.env_map.data
    w = env_light.solid_angle_weight  # [1,H,W]
    lum = 0.2126 * env[0:1] + 0.7152 * env[1:2] + 0.0722 * env[2:3]
    mean_lum = float((lum * w).sum().item() / (w.sum().item() + 1e-8))
    if not np.isfinite(mean_lum) or mean_lum <= 1e-8:
        return
    scale = target_mean / mean_lum
    env.mul_(float(scale))


def _maybe_upsample_env_light(args, env_light: EnvironmentLight, new_res: int) -> EnvironmentLight:
    if int(new_res) <= 0 or int(new_res) == int(env_light.resolution):
        return env_light

    with torch.no_grad():
        old = env_light.env_map.detach().unsqueeze(0)  # [1,3,H,2H]
        up = F.interpolate(old, size=(int(new_res), int(new_res) * 2), mode="bilinear", align_corners=False).squeeze(0)

    new_env = EnvironmentLight(env_map_path=None, resolution=int(new_res), num_mip_levels=env_light.num_mip_levels).cuda()
    with torch.no_grad():
        new_env.env_map.data.copy_(up)
    if not getattr(args, "no_env_gradient_scaling", False):
        new_env.register_gradient_scaling_hook()
    print(f"Upsampled env_map: {env_light.resolution} -> {new_env.resolution}")
    return new_env


def train_env_light(dataset, opt, pipe, args) -> None:
    prepare_output(args)
    safe_state(getattr(args, "quiet", False))

    # 1) Fixed geometry from dense GT point cloud
    pcd = fetchPly(args.gt_ply)
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=True)
    gaussians.roughness_min = float(getattr(args, "roughness_min", 0.02))
    gaussians.roughness_max = float(getattr(args, "roughness_max", 0.999))
    gaussians.create_from_dense_pcd(pcd, spatial_lr_scale=1.0)

    # Optional: initialize roughness to a specific value, then freeze during stage 1.
    roughness_init = getattr(args, "roughness_init", None)
    if roughness_init is not None:
        r = float(roughness_init)
        r = max(1e-6, min(1.0 - 1e-6, r))
        with torch.no_grad():
            gaussians._roughness.data.fill_(float(inverse_sigmoid(torch.tensor(r)).item()))
        print(f"Initialized roughness to {r}")

    # 2) Scene (cameras only; gaussians already initialized)
    scene = Scene(dataset, gaussians)
    gaussians.spatial_lr_scale = scene.cameras_extent

    # 3) Optimizers (static geometry: no xyz/rot)
    gaussians.training_setup_fixed_geometry_pbr_only(opt)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    dummy_color = torch.zeros((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")

    # 4) Environment light (optionally start lower-res for stage 1)
    env_res = int(getattr(args, "env_map_res", 256))
    env_res_stage1 = getattr(args, "env_map_res_stage1", None)
    if env_res_stage1 is not None:
        env_res_stage1 = int(env_res_stage1)
    env_light = EnvironmentLight(args.env_map, resolution=(env_res_stage1 or env_res)).cuda()
    if not args.no_env_gradient_scaling:
        env_light.register_gradient_scaling_hook()
    env_opt = torch.optim.Adam(env_light.parameters(), lr=float(args.env_light_lr))

    stage1_iters = int(getattr(args, "stage1_iters", 2000))
    alt_env_steps = int(getattr(args, "alt_env_steps", 5))
    alt_mat_steps = int(getattr(args, "alt_mat_steps", 1))
    cycle = max(1, alt_env_steps + alt_mat_steps)

    progress = tqdm(range(1, int(args.iterations) + 1), desc="Pretraining EnvLight (object-only)")
    viewpoint_stack = None

    for it in progress:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if cam.gt_alpha_mask is None:
            raise RuntimeError("Foreground mask (gt_alpha_mask) is required for object-only env pretraining.")

        mask = _process_mask(
            cam.gt_alpha_mask.to("cuda"),
            binarize=bool(getattr(args, "mask_binarize", False)),
            threshold=float(getattr(args, "mask_threshold", 0.5)),
        )
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)

        # Stage boundary: upsample envmap once after stage 1
        if it == stage1_iters + 1 and env_res_stage1 is not None and env_res_stage1 != env_res:
            env_light = _maybe_upsample_env_light(args, env_light, env_res)
            env_opt = torch.optim.Adam(env_light.parameters(), lr=float(args.env_light_lr))

        # Render fixed-geometry buffers
        render_pkg = render(cam, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)
        alpha = render_pkg["rend_alpha"]
        denom = alpha + 1e-6

        albedo = torch.clamp(render_pkg["gbuffer_albedo"] / denom, 0.0, 1.0)
        rough = torch.clamp(render_pkg["gbuffer_roughness"] / denom, float(gaussians.roughness_min), float(gaussians.roughness_max))
        metal = torch.clamp(render_pkg["gbuffer_metallic"] / denom, 0.0, 1.0)
        normal = render_pkg["rend_normal"] / denom
        depth = render_pkg.get("surf_depth")

        ray_dirs = _get_ray_dirs_world(cam)
        shaded_obj = screen_space_pbr_shading(
            albedo,
            rough,
            metal,
            normal,
            depth,
            cam.camera_center,
            cam.world_view_transform,
            env_light=env_light,
            ray_dirs_world=ray_dirs,
        )

        # Object-only reconstruction with highlight robustness.
        weight = mask
        q = getattr(args, "highlight_quantile", None)
        if (q is not None) and (0.0 < float(q) < 1.0) and (float(getattr(args, "highlight_drop", 0.0)) > 0.0):
            weight = _highlight_weight_map(
                gt_rgb=gt,
                obj_mask=mask,
                q=float(q),
                drop=float(getattr(args, "highlight_drop", 0.0)),
            )

        Ll1 = l1_loss(shaded_obj, gt, mask=weight)
        ssim_val = ssim(shaded_obj.unsqueeze(0), gt.unsqueeze(0), mask=weight.unsqueeze(0))
        recon = (1.0 - float(args.lambda_dssim)) * Ll1 + float(args.lambda_dssim) * (1.0 - ssim_val)

        # Strong material regularization (screen-space) to keep materials from absorbing lighting.
        pbr_reg = torch.tensor(0.0, device="cuda")
        if it > stage1_iters:
            pbr_losses = compute_pbr_losses(
                albedo,
                rough,
                metal,
                alpha_map=mask,
                lambda_albedo_smooth=float(args.lambda_albedo_smooth),
                lambda_roughness_smooth=float(args.lambda_roughness_smooth),
                lambda_metallic_smooth=float(args.lambda_metallic_smooth),
                lambda_metallic_prior=float(args.lambda_metallic_prior),
                lambda_roughness_prior=float(args.lambda_roughness_prior),
                lambda_albedo_chroma=float(args.lambda_albedo_chroma),
            )
            pbr_reg = float(args.lambda_pbr_reg) * pbr_losses["total_pbr_reg"]

        env_tv = float(getattr(args, "lambda_env_tv", 0.0)) * env_light.tv_loss_weighted()
        env_smooth = float(getattr(args, "lambda_env_smooth", 0.0)) * env_light.smoothness_loss_weighted()
        loss = float(args.lambda_rgb) * recon + pbr_reg + env_tv + env_smooth

        gaussians.optimizer.zero_grad(set_to_none=True)
        env_opt.zero_grad(set_to_none=True)
        loss.backward()

        # Step schedule
        if it <= stage1_iters:
            do_env = True
            do_mat = False
        else:
            phase = (it - stage1_iters - 1) % cycle
            do_env = phase < alt_env_steps
            do_mat = not do_env

        if do_env:
            env_opt.step()
            target_mean = getattr(args, "env_gauge_target_mean_luminance", None)
            if target_mean is not None:
                _gauge_normalize_env_mean_luminance(env_light, float(target_mean))
            if (args.env_clamp_min is not None) or (args.env_clamp_max is not None):
                with torch.no_grad():
                    min_v = float(args.env_clamp_min) if args.env_clamp_min is not None else -float("inf")
                    max_v = float(args.env_clamp_max) if args.env_clamp_max is not None else float("inf")
                    env_light.env_map.data.clamp_(min=min_v, max=max_v)

        if do_mat:
            gaussians.optimizer.step()

        if it % int(getattr(args, "log_interval", 50)) == 0:
            progress.set_postfix(
                loss=float(loss.detach().cpu().item()),
                recon=float(recon.detach().cpu().item()),
                pbr_reg=float(pbr_reg.detach().cpu().item()),
                env_mean=float(env_light.env_map.detach().mean().cpu().item()),
                do_env=int(do_env),
                do_mat=int(do_mat),
            )

        if args.save_interval and (it % int(args.save_interval) == 0):
            torch.save(env_light.state_dict(), os.path.join(args.model_path, f"env_light_iter_{it:06d}.pth"))

    out_path = os.path.join(args.model_path, "env_light_final.pth")
    torch.save(env_light.state_dict(), out_path)
    print(f"Saved env_light: {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone env_light pretraining (object-only)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--gt_ply", type=str, required=True, help="Dense GT point cloud PLY (with normals recommended)")

    # Env light
    parser.add_argument("--env_map", type=str, default=None, help="Optional initial env map (.hdr/.exr/.png)")
    parser.add_argument("--env_light_lr", type=float, default=1e-2)
    parser.add_argument("--env_map_res", type=int, default=256)
    parser.add_argument("--env_map_res_stage1", type=int, default=None, help="Optional lower resolution during stage 1")
    parser.add_argument("--no_env_gradient_scaling", action="store_true")
    parser.add_argument("--env_gauge_target_mean_luminance", type=float, default=0.5, help="Gauge fix: target mean luminance (<=0 disables)")
    parser.add_argument("--env_clamp_min", type=float, default=None)
    parser.add_argument("--env_clamp_max", type=float, default=None)
    parser.add_argument("--lambda_env_tv", type=float, default=0.0)
    parser.add_argument("--lambda_env_smooth", type=float, default=0.0)

    # Reconstruction
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--mask_binarize", action="store_true")
    parser.add_argument("--mask_threshold", type=float, default=0.5)

    # Highlight robustness (object-only)
    parser.add_argument("--highlight_quantile", type=float, default=0.98, help="Quantile for highlight pixels in object mask")
    parser.add_argument("--highlight_drop", type=float, default=0.0, help="Downweight factor for highlight pixels (0 disables, 1 drops)")

    # Stage schedule
    parser.add_argument("--stage1_iters", type=int, default=2000)
    parser.add_argument("--alt_env_steps", type=int, default=5)
    parser.add_argument("--alt_mat_steps", type=int, default=1)

    # Material regularization (strong)
    parser.add_argument("--lambda_pbr_reg", type=float, default=0.1)
    parser.add_argument("--lambda_albedo_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_roughness_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_metallic_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_metallic_prior", type=float, default=0.01)
    parser.add_argument("--lambda_roughness_prior", type=float, default=0.01)
    parser.add_argument("--lambda_albedo_chroma", type=float, default=0.01)
    parser.add_argument("--roughness_init", type=float, default=0.3)
    parser.add_argument("--roughness_min", type=float, default=0.02)
    parser.add_argument("--roughness_max", type=float, default=0.999)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    # For pretraining, prefer conservative material LRs; user can override via standard OptimizationParams flags.
    train_env_light(dataset, opt, pipe, args)
