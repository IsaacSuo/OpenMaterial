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
from utils.general_utils import safe_state
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
        
        # Loss computation
        # Standard reconstruction loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
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
            
            pbr_loss = opt.lambda_pbr * pbr_reconstruction_loss(shaded_image, gt_image)
            env_tv_loss = opt.lambda_env_tv * env_light.tv_loss_weighted()
            
            # PBR Regularization
            pbr_losses = compute_pbr_losses(gbuffer_albedo, gbuffer_roughness, gbuffer_metallic, alpha_map)
            pbr_reg_loss = opt.lambda_pbr_reg * pbr_losses['total_pbr_reg']

        total_loss = loss + pbr_loss + env_tv_loss + pbr_reg_loss
        
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
    parser.add_argument("--lambda_pbr", type=float, default=0.1)
    parser.add_argument("--lambda_pbr_reg", type=float, default=0.01)
    parser.add_argument("--env_light_lr", type=float, default=0.01)
    parser.add_argument("--lambda_env_tv", type=float, default=0.001)
    parser.add_argument("--no_env_gradient_scaling", action="store_true")
    
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
    opt.env_light_lr = args.env_light_lr
    opt.lambda_pbr = args.lambda_pbr
    opt.lambda_pbr_reg = args.lambda_pbr_reg
    opt.lambda_env_tv = args.lambda_env_tv
    
training_pbr_static(lp.extract(args), opt, pp.extract(args), args)
