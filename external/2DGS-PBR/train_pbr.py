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


def scale_invariant_loss(pred, gt):
    """
    Solve for s, t such that || s * pred + t - gt || is minimized,
    then return the L1 loss of the aligned prediction.
    """
    mask = (gt > 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")
    
    p = pred[mask]
    g = gt[mask]
    
    # Linear regression: p * s + t = g
    ones = torch.ones_like(p)
    A = torch.stack([p, ones], dim=1) # [N, 2]
    
    try:
        X, _ = torch.linalg.lstsq(A, g).solution
        s, t = X[0], X[1]
        
        # Prevent negative scale if physically implausible (optional but recommended)
        if s < 0: s = torch.tensor(0.0, device="cuda")
        
        pred_aligned = pred * s + t
        return torch.nn.functional.l1_loss(pred_aligned[mask], g)
    except:
        # Fallback if SVD fails
        return torch.tensor(0.0, device="cuda")


def training_pbr(dataset, opt, pipe, testing_iterations, saving_iterations,
                 checkpoint_iterations, checkpoint, env_map_path=None):
    """
    PBR-enhanced training loop.
    """
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

    # Environment light regularization weight
    lambda_env_tv = getattr(opt, 'lambda_env_tv', 0.001)

    # Geometric supervision weights
    use_pseudo_gt = getattr(opt, 'use_pseudo_gt', False)
    lambda_mono_depth = getattr(opt, 'lambda_mono_depth', 0.1)
    lambda_mono_normal = getattr(opt, 'lambda_mono_normal', 0.05)
    
    # Pre-construct GT paths
    if use_pseudo_gt:
        depth_root = os.path.join(dataset.source_path, getattr(opt, 'depth_subdir', 'depth'))
        normal_root = os.path.join(dataset.source_path, getattr(opt, 'normal_subdir', 'normal'))
        print(f"Geometric Supervision Enabled.")
        print(f"Depth GT path: {depth_root}")
        print(f"Normal GT path: {normal_root}")
    else:
        depth_root = None
        normal_root = None

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

    # PBR loss weights
    lambda_pbr = getattr(opt, 'lambda_pbr', 0.1)
    lambda_pbr_reg = getattr(opt, 'lambda_pbr_reg', 0.01)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training PBR")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_pbr=True)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # Standard reconstruction loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # PBR shading loss
        pbr_loss = torch.tensor(0.0, device="cuda")
        pbr_reg_loss = torch.tensor(0.0, device="cuda")
        env_tv_loss = torch.tensor(0.0, device="cuda")
        loss_mono_depth = torch.tensor(0.0, device="cuda")
        loss_mono_normal = torch.tensor(0.0, device="cuda")

        if iteration > 5000 and gaussians.use_pbr:
            # Get G-Buffer
            gbuffer_albedo = render_pkg.get('gbuffer_albedo')
            gbuffer_roughness = render_pkg.get('gbuffer_roughness')
            gbuffer_metallic = render_pkg.get('gbuffer_metallic')
            gbuffer_normal = render_pkg.get('rend_normal')
            gbuffer_depth = render_pkg.get('surf_depth')
            alpha_map = render_pkg.get('rend_alpha')

            if gbuffer_albedo is not None:
                # Apply PBR shading
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
                pbr_loss = lambda_pbr * pbr_reconstruction_loss(shaded_image, gt_image)

                # Material regularization losses
                if iteration > 10000:
                    pbr_losses = compute_pbr_losses(
                        gbuffer_albedo,
                        gbuffer_roughness,
                        gbuffer_metallic,
                        alpha_map,
                    )
                    pbr_reg_loss = lambda_pbr_reg * pbr_losses['total_pbr_reg']

                # Environment light TV regularization
                env_tv_loss = lambda_env_tv * env_light.tv_loss_weighted()

        # === Geometric Supervision (Pseudo-GT / Real-GT) ===
        if use_pseudo_gt:
            # Load GT
            gt_depth, gt_normal = load_pseudo_gt(viewpoint_cam, depth_root, normal_root)
            
            # 1. Depth Supervision
            if gt_depth is not None:
                pred_depth = render_pkg["surf_depth"]
                # Use scale-invariant loss
                loss_mono_depth = lambda_mono_depth * scale_invariant_loss(pred_depth, gt_depth)

            # 2. Normal Supervision
            if gt_normal is not None:
                # Use rendered normal (rend_normal) or geometric normal (surf_normal)
                # rend_normal is better as it comes from Gaussian rotation
                pred_normal = render_pkg["rend_normal"]
                
                # Normalize both
                pred_norm = F.normalize(pred_normal, dim=0)
                gt_norm = F.normalize(gt_normal, dim=0)
                
                # Cosine Similarity Loss: 1 - cos(theta)
                cosine_sim = (pred_norm * gt_norm).sum(dim=0)
                # Only supervise valid pixels (where normal is not zero or masked)
                # Assuming GT normal 0,0,0 is invalid
                valid_mask = (gt_normal.abs().sum(dim=0) > 0.1)
                
                if valid_mask.sum() > 0:
                    loss_mono_normal = lambda_mono_normal * (1.0 - cosine_sim[valid_mask]).mean()

        # Geometry regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Total loss
        total_loss = loss + dist_loss + normal_loss + pbr_loss + pbr_reg_loss + env_tv_loss + loss_mono_depth + loss_mono_normal

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

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "pbr": f"{ema_pbr_for_log:.{4}f}",
                    "env": f"{ema_env_tv_for_log:.{4}f}",
                    "m_d": f"{ema_mono_depth_for_log:.{4}f}",
                    "m_n": f"{ema_mono_normal_for_log:.{4}f}",
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
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # Environment light optimizer (only after PBR training starts)
                if iteration > 5000:
                    env_light_optimizer.step()
                    env_light_optimizer.zero_grad(set_to_none=True)

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

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

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

    training_pbr(
        lp.extract(args), opt, pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        env_map_path=args.env_map
    )

    print("\nPBR Training complete.")
