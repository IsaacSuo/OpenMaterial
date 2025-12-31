#!/usr/bin/env python3
"""
PBR Training Script for 2DGS-PBR

Extends the standard 2DGS training with:
- PBR material parameters (albedo, roughness, metallic)
- G-Buffer deferred rendering
- PBR shading with environment lighting
- Material regularization losses
"""

import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_pbr_losses, pbr_reconstruction_loss
from utils.pbr_utils import EnvironmentLight, screen_space_pbr_shading
from utils.loss_scheduler import LossScheduler, TrainingStages, LossWeights
from utils.profiler import SimpleProfiler
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def load_pseudo_gt(viewpoint_cam, depth_path_root, normal_path_root):
    """
    Load GT Depth and Normal maps based on the current viewpoint image name.
    """
    filename = viewpoint_cam.image_name
    
    gt_depth = None
    gt_normal = None
    
    # 1. Load Depth
    d_path = os.path.join(depth_path_root, f"{filename}.png")
    if os.path.exists(d_path):
        # Read as 16-bit unchanged (OpenCV returns uint16)
        depth_img = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            # Convert to float tensor
            depth_tensor = torch.from_numpy(depth_img.astype(np.float32)).cuda()
            
            # If using 16-bit PNG from our script, scale is 1000.0 (mm to m)
            # But since we use scale-invariant loss, the absolute scale doesn't matter much.
            # We can keep it as is or normalize.
            
            if depth_tensor.shape[:2] != (viewpoint_cam.image_height, viewpoint_cam.image_width):
                 depth_tensor = TF.resize(depth_tensor.unsqueeze(0), 
                                        (viewpoint_cam.image_height, viewpoint_cam.image_width))
                 depth_tensor = depth_tensor.squeeze(0)
            
            gt_depth = depth_tensor.unsqueeze(0) # [1, H, W]

    # 2. Load Normal
    n_path = os.path.join(normal_path_root, f"{filename}.png")
    if os.path.exists(n_path):
        normal_img = cv2.imread(n_path)
        if normal_img is not None:
            # OpenCV is BGR, convert to RGB
            normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
            normal_tensor = torch.from_numpy(normal_img.astype(np.float32) / 255.0).cuda()
            
            # Map [0, 1] back to [-1, 1]
            normal_tensor = normal_tensor * 2.0 - 1.0
            
            # [H, W, 3] -> [3, H, W]
            normal_tensor = normal_tensor.permute(2, 0, 1)
            
            if normal_tensor.shape[1:] != (viewpoint_cam.image_height, viewpoint_cam.image_width):
                normal_tensor = TF.resize(normal_tensor, 
                                        (viewpoint_cam.image_height, viewpoint_cam.image_width))
            
            gt_normal = normal_tensor

    return gt_depth, gt_normal


def scale_invariant_loss(pred, gt, debug=False):
    """
    Solve for s, t such that || s * pred + t - gt || is minimized,
    then return the L1 loss of the aligned prediction.

    Optimized: Solves linear regression on downsampled data for speed,
    then applies the solution to full resolution.
    """
    mask = (gt > 0)
    if debug:
        print(f"[DEBUG] gt range: {gt.min():.2f} ~ {gt.max():.2f}, mask.sum(): {mask.sum()}")
    if mask.sum() == 0:
        if debug:
            print("[DEBUG] mask.sum() == 0, returning 0")
        return torch.tensor(0.0, device="cuda")

    # Optimization: Solve system on downsampled data for speed
    # lstsq is O(N^2), so 1/4 downsample reduces computation ~16x
    if pred.numel() > 64 * 64:
        scale_factor = 0.25
        p_small = F.interpolate(pred.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
        g_small = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, mode='nearest').squeeze(0)
        mask_small = (g_small > 0)

        if mask_small.sum() > 10:  # Ensure enough points for regression
            p_fit = p_small[mask_small]
            g_fit = g_small[mask_small]
        else:
            p_fit = pred[mask]
            g_fit = gt[mask]
    else:
        p_fit = pred[mask]
        g_fit = gt[mask]

    # Linear regression: p * s + t = g
    ones = torch.ones_like(p_fit)
    A = torch.stack([p_fit, ones], dim=1)

    try:
        result = torch.linalg.lstsq(A, g_fit)
        solution = result.solution  # [2] tensor: [s, t]
        if debug:
            print(f"[DEBUG] lstsq solution shape: {solution.shape}, values: {solution}")
        s, t = solution[0], solution[1]

        # Check for nan/inf from lstsq (numerical instability)
        if torch.isnan(s) or torch.isnan(t) or torch.isinf(s) or torch.isinf(t):
            if debug:
                print(f"[DEBUG] lstsq returned nan/inf: s={s}, t={t}")
            return torch.tensor(0.0, device="cuda")

        # Prevent negative scale if physically implausible
        if s < 0:
            if debug:
                print(f"[DEBUG] s < 0 ({s:.4f}), setting to 0")
            s = torch.tensor(0.0, device="cuda")

        # Apply to full resolution
        pred_aligned = pred * s + t

        # Compute relative L1 error (normalized by GT mean) to keep loss scale-invariant
        gt_mean = gt[mask].mean()
        loss = torch.nn.functional.l1_loss(pred_aligned[mask], gt[mask]) / (gt_mean + 1e-8)

        # Final nan check
        if torch.isnan(loss):
            if debug:
                print(f"[DEBUG] loss is nan, s={s}, t={t}, gt_mean={gt_mean}")
            return torch.tensor(0.0, device="cuda")

        if debug:
            print(f"[DEBUG] s={s:.4f}, t={t:.4f}, gt_mean={gt_mean:.2f}, loss={loss:.6f}")
        return loss
    except Exception as e:
        # Fallback if lstsq fails (e.g., singular matrix)
        print(f"[Warning] scale_invariant_loss failed: {e}")
        return torch.tensor(0.0, device="cuda")


def training_pbr(dataset, opt, pipe, testing_iterations, saving_iterations,
                 checkpoint_iterations, checkpoint, env_map_path=None,
                 profiler=None):
    """
    PBR-enhanced training loop.
    """
    if profiler is None:
        profiler = SimpleProfiler(enabled=False)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Create Gaussian model with PBR enabled
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=True)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load environment light (learnable)
    env_light = EnvironmentLight(env_map_path, resolution=256).cuda()

    # Environment light optimizer
    env_light_lr = getattr(opt, 'env_light_lr', 0.01)
    env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=env_light_lr)
    print(f"Environment light: learnable, resolution=256, lr={env_light_lr}")

    # Optional: Register gradient scaling hook for solid-angle weighted optimization
    use_env_gradient_scaling = getattr(opt, 'env_gradient_scaling', True)
    if use_env_gradient_scaling:
        env_light.register_gradient_scaling_hook()

    # Create loss scheduler
    loss_scheduler = LossScheduler.from_args(opt)
    loss_scheduler.print_config()

    # Geometric supervision
    use_pseudo_gt = getattr(opt, 'use_pseudo_gt', False)

    # Preload pseudo-GT data if enabled (eliminates per-iteration I/O)
    if use_pseudo_gt:
        depth_subdir = getattr(opt, 'depth_subdir', 'depth')
        normal_subdir = getattr(opt, 'normal_subdir', 'normal')
        scene.load_pseudo_gt_cache(depth_subdir, normal_subdir)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_pbr_for_log = 0.0
    ema_env_tv_for_log = 0.0
    ema_mono_depth_for_log = 0.0
    ema_mono_normal_for_log = 0.0
    ema_alpha_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training PBR")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        profiler.start_iteration()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render with G-Buffer for PBR
        with profiler.profile("render"):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_pbr=True)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # Get mask if available
        gt_mask = viewpoint_cam.gt_alpha_mask
        if gt_mask is not None:
            gt_mask = gt_mask.cuda()

        # Standard reconstruction loss (with mask if available)
        Ll1 = l1_loss(image, gt_image, gt_mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=gt_mask))

        # Get scheduled loss weights for current iteration
        weights = loss_scheduler.get_weights(iteration)

        # Initialize loss terms
        pbr_loss = torch.tensor(0.0, device="cuda")
        pbr_reg_loss = torch.tensor(0.0, device="cuda")
        env_tv_loss = torch.tensor(0.0, device="cuda")
        loss_mono_depth = torch.tensor(0.0, device="cuda")
        loss_mono_normal = torch.tensor(0.0, device="cuda")
        loss_alpha = torch.tensor(0.0, device="cuda")

        # === Geometry Regularization (Stage 2+) ===
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

        dist_loss = weights['dist'] * rend_dist.mean()
        normal_loss = weights['normal'] * normal_error.mean()

        # === Alpha Supervision ===
        # Supervise rend_alpha to match gt_mask
        if gt_mask is not None and opt.lambda_alpha > 0:
            rend_alpha = render_pkg["rend_alpha"]
            loss_alpha = opt.lambda_alpha * l1_loss(rend_alpha, gt_mask)

        # === Mono Supervision (Stage 2+, fade in Stage 4) ===
        if use_pseudo_gt and (weights['mono_depth'] > 0 or weights['mono_normal'] > 0):
            gt_depth = viewpoint_cam.pseudo_gt_depth
            gt_normal = viewpoint_cam.pseudo_gt_normal

            if gt_depth is not None and weights['mono_depth'] > 0:
                with profiler.profile("mono_depth_transfer"):
                    gt_depth = gt_depth.cuda()
                pred_depth = render_pkg["surf_depth"]
                # Debug first few iterations after mono starts
                debug_depth = (iteration >= loss_scheduler.stages.stage1_end and
                               iteration < loss_scheduler.stages.stage1_end + 5)
                with profiler.profile("mono_depth_lstsq"):
                    loss_mono_depth = weights['mono_depth'] * scale_invariant_loss(pred_depth, gt_depth, debug=debug_depth)

            if gt_normal is not None and weights['mono_normal'] > 0:
                with profiler.profile("mono_normal_transfer"):
                    gt_normal = gt_normal.cuda()
                pred_normal = render_pkg["rend_normal"]

                with profiler.profile("mono_normal_loss"):
                    pred_norm = F.normalize(pred_normal, dim=0)
                    gt_norm = F.normalize(gt_normal, dim=0)
                    cosine_sim = (pred_norm * gt_norm).sum(dim=0)
                    valid_mask = (torch.norm(gt_normal, dim=0) > 0.1)

                    # Debug: print normal statistics every 500 iterations
                    if iteration % 500 == 0 and valid_mask.sum() > 0:
                        print(f"\n[Normal Debug @ iter {iteration}]")
                        print(f"  valid_mask: {valid_mask.sum()}/{valid_mask.numel()}")
                        print(f"  GT normal mean: {gt_norm[:, valid_mask].mean(dim=1).tolist()}")
                        print(f"  Pred normal mean: {pred_norm[:, valid_mask].mean(dim=1).tolist()}")
                        print(f"  cosine_sim: min={cosine_sim[valid_mask].min():.3f}, max={cosine_sim[valid_mask].max():.3f}, mean={cosine_sim[valid_mask].mean():.3f}")

                    if valid_mask.sum() > 0:
                        loss_mono_normal = weights['mono_normal'] * (1.0 - cosine_sim[valid_mask]).mean()

        # === PBR Shading (Stage 3+) ===
        if loss_scheduler.is_pbr_active(iteration) and gaussians.use_pbr:
            gbuffer_albedo = render_pkg.get('gbuffer_albedo')
            gbuffer_roughness = render_pkg.get('gbuffer_roughness')
            gbuffer_metallic = render_pkg.get('gbuffer_metallic')
            gbuffer_normal = render_pkg.get('rend_normal')
            gbuffer_depth = render_pkg.get('surf_depth')
            alpha_map = render_pkg.get('rend_alpha')

            if gbuffer_albedo is not None:
                # PBR shading
                with profiler.profile("pbr_shading"):
                    shaded_image = screen_space_pbr_shading(
                        gbuffer_albedo,
                        gbuffer_roughness,
                        gbuffer_metallic,
                        gbuffer_normal,
                        gbuffer_depth,
                        viewpoint_cam.camera_center,
                        viewpoint_cam.world_view_transform,
                        env_light=env_light,
                    )

                # PBR reconstruction loss
                pbr_loss = weights['pbr'] * pbr_reconstruction_loss(shaded_image, gt_image)

                # Environment light TV regularization
                env_tv_loss = weights['env_tv'] * env_light.tv_loss_weighted()

                # === PBR Regularization (Stage 4) ===
                if loss_scheduler.is_pbr_reg_active(iteration):
                    with profiler.profile("pbr_reg"):
                        pbr_losses = compute_pbr_losses(
                            gbuffer_albedo,
                            gbuffer_roughness,
                            gbuffer_metallic,
                            alpha_map,
                        )
                        pbr_reg_loss = weights['pbr_reg'] * pbr_losses['total_pbr_reg']

        # Total loss
        total_loss = loss + dist_loss + normal_loss + pbr_loss + pbr_reg_loss + env_tv_loss + loss_mono_depth + loss_mono_normal + loss_alpha

        with profiler.profile("backward"):
            total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_pbr_for_log = 0.4 * (pbr_loss.item() + pbr_reg_loss.item()) + 0.6 * ema_pbr_for_log
            ema_env_tv_for_log = 0.4 * env_tv_loss.item() + 0.6 * ema_env_tv_for_log
            ema_mono_depth_for_log = 0.4 * loss_mono_depth.item() + 0.6 * ema_mono_depth_for_log
            ema_mono_normal_for_log = 0.4 * loss_mono_normal.item() + 0.6 * ema_mono_normal_for_log
            ema_alpha_for_log = 0.4 * loss_alpha.item() + 0.6 * ema_alpha_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "pbr": f"{ema_pbr_for_log:.{4}f}",
                    "env": f"{ema_env_tv_for_log:.{4}f}",
                    "m_d": f"{ema_mono_depth_for_log:.{4}f}",
                    "m_n": f"{ema_mono_normal_for_log:.{4}f}",
                    "alpha": f"{ema_alpha_for_log:.{4}f}",
                    "pts": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/pbr_loss', ema_pbr_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/env_tv_loss', ema_env_tv_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/mono_depth_loss', ema_mono_depth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/mono_normal_loss', ema_mono_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/alpha_loss', ema_alpha_for_log, iteration)

            training_report_pbr(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background),
                env_light
            )


            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # Save environment light
                env_light_path = os.path.join(scene.model_path, f"env_light_{iteration}.pth")
                torch.save(env_light.state_dict(), env_light_path)
                print(f"[ITER {iteration}] Saved environment light to {env_light_path}")

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull,
                        scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                with profiler.profile("optimizer_step"):
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                    # Environment light optimizer (Stage 3+: when PBR is active)
                    if loss_scheduler.should_optimize_env_light(iteration):
                        env_light_optimizer.step()
                        env_light_optimizer.zero_grad(set_to_none=True)

            profiler.end_iteration()
            if iteration % 500 == 0:
                profiler.report(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )

        # Network GUI handling
        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255).byte()
                            .permute(1, 2, 0).contiguous().cpu().numpy()
                        )
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None


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


@torch.no_grad()
def training_report_pbr(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                        testing_iterations, scene, renderFunc, renderArgs, env_light):
    """Extended training report with PBR visualization."""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ]}
        )

        pipe, background = renderArgs

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, background, render_pbr=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap

                        # Depth visualization
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth_vis = depth / norm if norm > 0 else depth
                        depth_vis = colormap(depth_vis.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(
                            config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                            depth_vis[None], global_step=iteration
                        )
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration
                        )

                        # PBR material visualization
                        if scene.gaussians.use_pbr:
                            gbuffer_albedo = render_pkg.get('gbuffer_albedo')
                            gbuffer_roughness = render_pkg.get('gbuffer_roughness')
                            gbuffer_metallic = render_pkg.get('gbuffer_metallic')

                            if gbuffer_albedo is not None:
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/albedo".format(viewpoint.image_name),
                                    gbuffer_albedo[None], global_step=iteration
                                )
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/roughness".format(viewpoint.image_name),
                                    gbuffer_roughness.expand(3, -1, -1)[None],
                                    global_step=iteration
                                )
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/metallic".format(viewpoint.image_name),
                                    gbuffer_metallic.expand(3, -1, -1)[None],
                                    global_step=iteration
                                )

                                # PBR shaded image
                                gbuffer_normal = render_pkg.get('rend_normal')
                                gbuffer_depth = render_pkg.get('surf_depth')
                                shaded = screen_space_pbr_shading(
                                    gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
                                    gbuffer_normal, gbuffer_depth,
                                    viewpoint.camera_center, viewpoint.world_view_transform,
                                    env_light=env_light
                                )
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/pbr_shaded".format(viewpoint.image_name),
                                    shaded[None], global_step=iteration
                                )

                        # Normal visualization
                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                rend_normal[None], global_step=iteration
                            )
                            tb_writer.add_images(
                                config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                surf_normal[None], global_step=iteration
                            )
                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                rend_alpha[None], global_step=iteration
                            )
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration
                            )

                    # Get mask if available for evaluation
                    gt_mask = viewpoint.gt_alpha_mask
                    if gt_mask is not None:
                        gt_mask = gt_mask.cuda()

                    l1_test += l1_loss(image, gt_image, gt_mask).mean().double()
                    psnr_test += psnr(image, gt_image, gt_mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test
                ))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="PBR Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # PBR-specific arguments
    parser.add_argument("--env_map", type=str, default=None, help="Path to HDR environment map (initial)")
    parser.add_argument("--lambda_pbr", type=float, default=0.1, help="PBR reconstruction loss weight")
    parser.add_argument("--lambda_pbr_reg", type=float, default=0.01, help="PBR regularization weight")
    parser.add_argument("--env_light_lr", type=float, default=0.01, help="Environment light learning rate")
    parser.add_argument("--lambda_env_tv", type=float, default=0.001,
                        help="Environment light TV regularization weight (solid-angle weighted)")
    parser.add_argument("--no_env_gradient_scaling", action="store_true", default=False,
                        help="Disable solid-angle gradient scaling for environment map")

    # Geometric Supervision arguments
    parser.add_argument("--use_pseudo_gt", action="store_true", default=False,
                        help="Enable geometric supervision using pseudo-GT depth/normal")
    parser.add_argument("--depth_subdir", type=str, default="depth",
                        help="Subdirectory name for depth GT images")
    parser.add_argument("--normal_subdir", type=str, default="normal",
                        help="Subdirectory name for normal GT images")
    parser.add_argument("--lambda_mono_depth", type=float, default=1.0,
                        help="Weight for mono-depth loss (relative error, so 1.0 is reasonable)")
    parser.add_argument("--lambda_mono_normal", type=float, default=0.05,
                        help="Weight for mono-normal loss")
    parser.add_argument("--lambda_alpha", type=float, default=0.1,
                        help="Weight for alpha supervision loss (rend_alpha vs gt_mask)")

    # Training Stage arguments
    parser.add_argument("--stage1_end", type=int, default=3000,
                        help="End of Stage 1 (geometry foundation, recon only)")
    parser.add_argument("--stage2_end", type=int, default=7000,
                        help="End of Stage 2 (+ dist, mono supervision)")
    parser.add_argument("--stage3_end", type=int, default=15000,
                        help="End of Stage 3 (+ normal, pbr, env_tv)")
    parser.add_argument("--warmup_iters", type=int, default=1000,
                        help="Warmup iterations for smooth loss activation")
    parser.add_argument("--fade_mono", action="store_true", default=True,
                        help="Fade out mono supervision in Stage 4")
    parser.add_argument("--no_fade_mono", action="store_true", default=False,
                        help="Disable fading of mono supervision")
    parser.add_argument("--fade_mono_end", type=int, default=20000,
                        help="Iteration when mono supervision is fully faded")

    # Profiling
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable performance profiling (prints timing every 500 iters)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing (PBR mode) " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Add PBR loss weights to optimization params
    opt = op.extract(args)
    opt.lambda_pbr = args.lambda_pbr
    opt.lambda_pbr_reg = args.lambda_pbr_reg
    opt.env_light_lr = args.env_light_lr
    opt.lambda_env_tv = args.lambda_env_tv
    opt.env_gradient_scaling = not args.no_env_gradient_scaling
    
    # Add geometric supervision params to opt
    opt.use_pseudo_gt = args.use_pseudo_gt
    opt.depth_subdir = args.depth_subdir
    opt.normal_subdir = args.normal_subdir
    opt.lambda_mono_depth = args.lambda_mono_depth
    opt.lambda_mono_normal = args.lambda_mono_normal

    # Add training stage params to opt
    opt.stage1_end = args.stage1_end
    opt.stage2_end = args.stage2_end
    opt.stage3_end = args.stage3_end
    opt.warmup_iters = args.warmup_iters
    opt.fade_mono = args.fade_mono and not args.no_fade_mono
    opt.fade_mono_end = args.fade_mono_end

    # Create profiler
    profiler = SimpleProfiler(enabled=args.profile)

    training_pbr(
        lp.extract(args), opt, pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        env_map_path=args.env_map,
        profiler=profiler
    )

    print("\nPBR Training complete.")
