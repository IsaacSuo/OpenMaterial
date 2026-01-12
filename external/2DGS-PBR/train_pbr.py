#!/usr/bin/env python3
"""
Static Geometry PBR Training Script for 2DGS-PBR

This script implements a "Static Geometry" pipeline where:
1. Geometry is initialized from a dense GT point cloud (with normals).
2. XYZ and Rotation are LOCKED (not optimized).
3. Only Scale, Opacity, SH, and PBR material parameters are optimized.
4. Densification (split/clone/prune) is DISABLED.
5. PBR shading and losses are enabled from the START.

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

from utils.loss_utils import l1_loss, ssim, compute_pbr_losses, pbr_reconstruction_loss
from utils.pbr_utils import EnvironmentLight, screen_space_pbr_shading
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
    gaussians.training_setup_fixed_geometry(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 4. Environment Light
    env_light = EnvironmentLight(args.env_map, resolution=256).cuda()
    env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=opt.env_light_lr)
    if not args.no_env_gradient_scaling:
        env_light.register_gradient_scaling_hook()

    # 5. Training Loop
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_pbr_for_log = 0.0
    
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

        # SH level up
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        # PBR is enabled from the start!
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_pbr=True)
        
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Get Mask for loss calculation
        mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else None
        
        # Loss computation
        # Standard reconstruction loss
        Ll1 = l1_loss(image, gt_image, mask=mask)
        
        # SSIM expects 4D input [B, C, H, W]
        image_4d = image.unsqueeze(0)
        gt_image_4d = gt_image.unsqueeze(0)
        mask_4d = mask.unsqueeze(0) if mask is not None else None
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_4d, gt_image_4d, mask=mask_4d))
        
        # PBR Shading Loss (Immediate activation)
        pbr_loss = torch.tensor(0.0, device="cuda")
        env_tv_loss = torch.tensor(0.0, device="cuda")
        pbr_reg_loss = torch.tensor(0.0, device="cuda")
        
        gbuffer_albedo = render_pkg.get('gbuffer_albedo')
        gbuffer_roughness = render_pkg.get('gbuffer_roughness')
        gbuffer_metallic = render_pkg.get('gbuffer_metallic')
        gbuffer_normal = render_pkg.get('rend_normal')
        gbuffer_depth = render_pkg.get('surf_depth')
        alpha_map = render_pkg.get('rend_alpha')

        if gbuffer_albedo is not None:
            shaded_image = screen_space_pbr_shading(
                gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
                gbuffer_normal, gbuffer_depth,
                viewpoint_cam.camera_center, viewpoint_cam.world_view_transform,
                env_light=env_light
            )
            
            pbr_loss = opt.lambda_pbr * pbr_reconstruction_loss(shaded_image, gt_image, mask=mask)
            env_tv_loss = opt.lambda_env_tv * env_light.tv_loss_weighted()
            
            # PBR Regularization
            # Use GT mask for regularization if available to focus on object surface
            reg_mask = mask if mask is not None else alpha_map
            pbr_losses = compute_pbr_losses(gbuffer_albedo, gbuffer_roughness, gbuffer_metallic, alpha_map=reg_mask)
            pbr_reg_loss = opt.lambda_pbr_reg * pbr_losses['total_pbr_reg']

        total_loss = opt.lambda_rgb * loss + pbr_loss + env_tv_loss + pbr_reg_loss
        
        # Backward
        total_loss.backward()
        iter_end.record()

        # Optimize
        with torch.no_grad():
            # Logging
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_pbr_for_log = 0.4 * pbr_loss.item() + 0.6 * ema_pbr_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "PBR": f"{ema_pbr_for_log:.{5}f}",
                    "Pts": f"{len(gaussians.get_xyz)}"
                })
                progress_bar.update(10)

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
                tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/pbr_loss', pbr_loss.item(), iteration)

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
                    {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 20, 5)]}
                )

                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        psnr_test = 0.0

                        for idx, viewpoint in enumerate(config['cameras']):
                            # Render with PBR enabled for visualization
                            render_pkg = render(viewpoint, gaussians, pipe, background, render_pbr=True)
                            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            # Log images only for the first few cameras to save TB space
                            if tb_writer and (idx < 4):
                                prefix = f"{config['name']}_view_{viewpoint.image_name}"
                                
                                # 1. Render vs GT
                                tb_writer.add_image(f"{prefix}/0_render", image, iteration)
                                if iteration == args.test_iterations[0]:
                                    tb_writer.add_image(f"{prefix}/0_gt", gt_image, iteration)

                                # 2. PBR Components
                                gbuffer_albedo = render_pkg.get('gbuffer_albedo')
                                if gbuffer_albedo is not None:
                                    # Albedo
                                    tb_writer.add_image(f"{prefix}/1_albedo", torch.clamp(gbuffer_albedo, 0, 1), iteration)
                                    
                                    # Roughness (Grayscale -> RGB for TB)
                                    rough = render_pkg.get('gbuffer_roughness')
                                    tb_writer.add_image(f"{prefix}/2_roughness", rough.repeat(3, 1, 1), iteration)
                                    
                                    # Metallic
                                    metal = render_pkg.get('gbuffer_metallic')
                                    tb_writer.add_image(f"{prefix}/3_metallic", metal.repeat(3, 1, 1), iteration)

                                    # 3. PBR Shaded Result
                                    # Important: use the current env_light for shading
                                    shaded = screen_space_pbr_shading(
                                        gbuffer_albedo, rough, metal,
                                        render_pkg.get('rend_normal'), render_pkg.get('surf_depth'),
                                        viewpoint.camera_center, viewpoint.world_view_transform,
                                        env_light=env_light
                                    )
                                    tb_writer.add_image(f"{prefix}/4_pbr_shaded", torch.clamp(shaded, 0, 1), iteration)

                                # 4. Geometry Check
                                rend_normal = render_pkg.get("rend_normal")
                                if rend_normal is not None:
                                    # Map [-1, 1] to [0, 1]
                                    norm_vis = torch.clamp(rend_normal * 0.5 + 0.5, 0, 1)
                                    tb_writer.add_image(f"{prefix}/5_normal", norm_vis, iteration)
                                
                                # 5. Depth
                                depth = render_pkg.get("surf_depth")
                                if depth is not None:
                                    d_max = depth.max()
                                    depth_norm = depth / d_max if d_max > 0 else depth
                                    depth_vis = colormap(depth_norm.cpu().numpy()[0], cmap='turbo') # [3, H, W]
                                    tb_writer.add_image(f"{prefix}/6_depth", depth_vis, iteration)

                            # Accumulate metrics
                            mask = viewpoint.gt_alpha_mask.cuda() if viewpoint.gt_alpha_mask is not None else None
                            
                            # Use masked metrics
                            l1_test += l1_loss(image, gt_image, mask=mask).mean().item()
                            psnr_test += psnr(image, gt_image, mask=mask).mean().item()

                        l1_test /= len(config['cameras'])
                        psnr_test /= len(config['cameras'])          
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
    parser.add_argument("--lambda_rgb", type=float, default=1.0, help="Weight for standard RGB reconstruction loss")
    parser.add_argument("--lambda_pbr", type=float, default=0.1)
    parser.add_argument("--lambda_pbr_reg", type=float, default=0.01)
    parser.add_argument("--env_light_lr", type=float, default=0.01)
    parser.add_argument("--lambda_env_tv", type=float, default=0.001)
    parser.add_argument("--no_env_gradient_scaling", action="store_true")

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
