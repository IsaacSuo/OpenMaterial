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
from argparse import ArgumentParser, Namespace
from random import randint
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim, compute_pbr_losses
from utils.pbr_utils import (
    EnvironmentLight,
    screen_space_pbr_shading,
    compute_ray_directions_world_from_fov,
)
from utils.profiler import SimpleProfiler
from utils.general_utils import safe_state, colormap
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

def _get_ray_dirs_world(viewpoint_cam) -> torch.Tensor:
    return compute_ray_directions_world_from_fov(
        image_height=viewpoint_cam.image_height,
        image_width=viewpoint_cam.image_width,
        fovx=viewpoint_cam.FoVx,
        fovy=viewpoint_cam.FoVy,
        world_view_transform=viewpoint_cam.world_view_transform,
        device="cuda",
    )

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

def training_pbr_static(dataset, opt, pipe, args):
    if args.gt_ply is None:
        raise ValueError("Error: --gt_ply argument is required for static geometry training.")

    tb_writer = prepare_output_and_logger(dataset)
    
    # 1. Initialize Gaussians from Dense PLY
    print(f"Loading dense GT point cloud from: {args.gt_ply}")
    pcd = fetchPly(args.gt_ply)
    
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=True)
    
    # Note: We use a placeholder spatial_lr_scale=1.0 initially.
    # It will be updated after Scene creation when we know the true extent.
    gaussians.create_from_dense_pcd(pcd, spatial_lr_scale=1.0)
    
    # 2. Initialize Scene
    # Since gaussians._xyz is now populated, Scene will NOT re-initialize them from COLMAP.
    scene = Scene(dataset, gaussians)
    
    # Update spatial_lr_scale with the correct extent from Scene
    gaussians.spatial_lr_scale = scene.cameras_extent
    print(f"Updated spatial_lr_scale to {scene.cameras_extent}")

    # 3. Setup Optimizer (Fixed Geometry Mode)
    # This locks XYZ and Rotation, but allows Scale, Opacity, and PBR to be optimized.
    gaussians.training_setup_fixed_geometry_pbr_only(opt)

    # PBR-only training does not supervise SH color output; keep background black so
    # G-buffer maps are clean premultiplied attributes (no background offset).
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    dummy_color = torch.zeros((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")

    # 4. Environment Light
    env_light = EnvironmentLight(args.env_map, resolution=256).cuda()
    env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=opt.env_light_lr)
    if not args.no_env_gradient_scaling:
        env_light.register_gradient_scaling_hook()

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
    
    first_iter = 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate (mainly for Opacity/SH, since XYZ is locked)
        gaussians.update_learning_rate(iteration)

        # No SH training in PBR-only mode.

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        # PBR is enabled from the start!
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)
        
        # Get Mask for loss calculation
        mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else None
        
        # --- Skybox / Background Rendering ---
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        ray_dirs = _get_ray_dirs_world(viewpoint_cam)
        bg_env = env_light.sample(ray_dirs).permute(2, 0, 1)
        
        gt_image = viewpoint_cam.original_image.cuda()

        # --- PBR + Skybox Composite Supervision (no SH render loss) ---
        alpha_map = render_pkg.get("rend_alpha")
        if alpha_map is None:
            raise RuntimeError("render_pkg missing 'rend_alpha'")

        gbuffer_albedo_pm = render_pkg.get("gbuffer_albedo")
        gbuffer_roughness_pm = render_pkg.get("gbuffer_roughness")
        gbuffer_metallic_pm = render_pkg.get("gbuffer_metallic")
        gbuffer_normal_pm = render_pkg.get("rend_normal")
        gbuffer_depth = render_pkg.get("surf_depth")
        if gbuffer_albedo_pm is None:
            raise RuntimeError("render_pbr=True but missing G-buffer outputs")

        # Unpremultiply (physical correctness requirement).
        eps = 1e-6
        denom = alpha_map + eps
        gbuffer_albedo = torch.clamp(gbuffer_albedo_pm / denom, 0.0, 1.0)
        gbuffer_roughness = torch.clamp(gbuffer_roughness_pm / denom, 0.1, 0.999)
        gbuffer_metallic = torch.clamp(gbuffer_metallic_pm / denom, 0.0, 1.0)
        gbuffer_normal = gbuffer_normal_pm / denom

        shaded_obj = screen_space_pbr_shading(
            gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
            gbuffer_normal, gbuffer_depth,
            viewpoint_cam.camera_center, viewpoint_cam.world_view_transform,
            env_light=env_light,
            ray_dirs_world=ray_dirs,
        )

        pred = shaded_obj * alpha_map + bg_env * (1.0 - alpha_map)

        # Composite supervision weights:
        # - Default: if gt_alpha_mask exists, supervise reconstruction only on the object region
        #   to avoid forcing env_light to match matted/black GT backgrounds.
        # - Opt-in: --supervise_background to supervise full composite (object + skybox).
        obj_mask = mask if mask is not None else alpha_map.detach()
        if (not getattr(args, "supervise_background", False)) and (mask is not None):
            recon_weight = mask
        else:
            recon_weight = torch.ones_like(alpha_map)
            if getattr(opt, "lambda_pbr", 0.0) > 0:
                recon_weight = recon_weight + opt.lambda_pbr * obj_mask

        env_tv_loss = opt.lambda_env_tv * env_light.tv_loss_weighted()

        reg_mask = mask if mask is not None else alpha_map.detach()
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

        # Full-image reconstruction loss (PBR object + skybox), with extra object-region weighting.
        Ll1 = l1_loss(pred, gt_image, mask=recon_weight)
        ssim_val = ssim(
            pred.unsqueeze(0),
            gt_image.unsqueeze(0),
            mask=recon_weight.unsqueeze(0),
        )
        recon_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        total_loss = opt.lambda_rgb * recon_loss + env_tv_loss + pbr_reg_loss
        
        # Backward
        total_loss.backward()
        iter_end.record()

        # Optimize
        with torch.no_grad():
            # Logging
            ema_total_for_log = 0.4 * total_loss.item() + 0.6 * ema_total_for_log
            ema_recon_for_log = 0.4 * recon_loss.item() + 0.6 * ema_recon_for_log
            ema_env_for_log = 0.4 * env_tv_loss.item() + 0.6 * ema_env_for_log
            ema_reg_for_log = 0.4 * pbr_reg_loss.item() + 0.6 * ema_reg_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Tot": f"{ema_total_for_log:.{5}f}",
                    "Recon": f"{ema_recon_for_log:.{5}f}",
                    "Env": f"{ema_env_for_log:.{2}e}",
                    "Reg": f"{ema_reg_for_log:.{2}e}",
                    "Pts": f"{len(gaussians.get_xyz)}"
                })
                progress_bar.update(10)

            if args.log_interval > 0 and (iteration % args.log_interval == 0):
                env_tv_unscaled = env_light.tv_loss_weighted().item()
                pbr_reg_unscaled = pbr_losses["total_pbr_reg"].item()
                obj_cov = (obj_mask > 0.5).float().mean().item()
                print(
                    f"\n[ITER {iteration}] total={total_loss.item():.6f} "
                    f"(lambda_rgb*recon={opt.lambda_rgb * recon_loss.item():.6f}, "
                    f"lambda_env_tv*tv={env_tv_loss.item():.6f}, "
                    f"lambda_pbr_reg*reg={pbr_reg_loss.item():.6f}) | "
                    f"recon={recon_loss.item():.6f} (L1={Ll1.item():.6f}, 1-SSIM={(1.0-ssim_val).item():.6f}) | "
                    f"tv_unscaled={env_tv_unscaled:.6e} reg_unscaled={pbr_reg_unscaled:.6e} | "
                    f"obj_cov={obj_cov:.3f} alpha_mean={alpha_map.mean().item():.3f} w_mean={recon_weight.mean().item():.3f}"
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
                torch.save(env_light.state_dict(), os.path.join(scene.model_path, f"env_light_{iteration}.pth"))

            # Step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            env_light_optimizer.step()
            env_light_optimizer.zero_grad(set_to_none=True)

            # --- Refined Evaluation and Image Logging ---
            if iteration in args.test_iterations:
                print(f"\n[ITER {iteration}] Running Evaluation...")
                torch.cuda.empty_cache()

                # We test all test cameras, and a few train cameras for consistency check
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

                    for cam_idx, viewpoint in enumerate(cameras):
                        render_pkg = render(viewpoint, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)

                        H, W = viewpoint.image_height, viewpoint.image_width
                        ray_dirs = _get_ray_dirs_world(viewpoint)
                        bg_env = env_light.sample(ray_dirs).permute(2, 0, 1)

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

                        if tb_writer and (cam_idx < 4):
                            prefix = f"{config['name']}_view_{viewpoint.image_name}"
                            tb_writer.add_image(f"{prefix}/0_render_composite", pred, iteration)
                            if iteration == args.test_iterations[0]:
                                tb_writer.add_image(f"{prefix}/0_gt", gt_image, iteration)
                            tb_writer.add_image(f"{prefix}/1_albedo", albedo, iteration)
                            tb_writer.add_image(f"{prefix}/2_roughness", rough.repeat(3, 1, 1), iteration)
                            tb_writer.add_image(f"{prefix}/3_metallic", metal.repeat(3, 1, 1), iteration)
                            tb_writer.add_image(f"{prefix}/4_pbr_shaded_obj", shaded, iteration)

                            rend_normal = render_pkg.get("rend_normal")
                            if rend_normal is not None:
                                norm_vis = torch.clamp(rend_normal * 0.5 + 0.5, 0, 1)
                                tb_writer.add_image(f"{prefix}/5_normal", norm_vis, iteration)

                            if depth_map is not None:
                                d_max = depth_map.max()
                                depth_norm = depth_map / d_max if d_max > 0 else depth_map
                                depth_vis = colormap(depth_norm.cpu().numpy()[0], cmap='turbo')
                                tb_writer.add_image(f"{prefix}/6_depth", depth_vis, iteration)

                        l1_test += l1_loss(pred, gt_image).mean().item()
                        psnr_test += psnr(pred, gt_image).mean().item()

                    l1_test /= len(cameras)
                    psnr_test /= len(cameras)
                    print(f"  [ITER {iteration}] {config['name']} PSNR: {psnr_test:.4f}")

                    if tb_writer:
                        tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
                        tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

                torch.cuda.empty_cache()

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
    parser.add_argument("--no_env_gradient_scaling", action="store_true")
    parser.add_argument(
        "--supervise_background",
        action="store_true",
        help="Supervise full composite (object + background). If unset and gt_alpha_mask exists, L1/SSIM is computed only on the mask region to avoid black-background supervision.",
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
    
    # Save/Test
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Print a detailed console loss breakdown every N iterations (0 disables)",
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Transfer args to opt
    opt = op.extract(args)
    opt.lambda_rgb = args.lambda_rgb
    opt.env_light_lr = args.env_light_lr
    opt.lambda_pbr = args.lambda_pbr
    opt.lambda_pbr_reg = args.lambda_pbr_reg
    opt.lambda_env_tv = args.lambda_env_tv
    
    training_pbr_static(lp.extract(args), opt, pp.extract(args), args)
