#!/usr/bin/env python3
"""
PBR Rendering Script for 2DGS-PBR

Renders test images with:
1. Standard SH-based rendering
2. PBR-shaded rendering (G-Buffer + environment lighting)
3. Material maps (albedo, roughness, metallic)

Usage:
    python render_pbr.py -m <model_path> --env_map <hdr_path>
"""

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pbr_utils import EnvironmentLight, screen_space_pbr_shading

import numpy as np
from PIL import Image


def save_image(tensor, path):
    """Save a [C, H, W] tensor as image"""
    img = tensor.detach().cpu().clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_single_channel(tensor, path, colormap=None):
    """Save a [1, H, W] tensor as grayscale or colormap image"""
    img = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    if colormap:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        img = (cmap(img)[:, :, :3] * 255).astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def render_set(dataset, iteration, pipeline, env_light, views, out_dir, split_name):
    """Render a set of views and save outputs"""

    makedirs(out_dir, exist_ok=True)

    # Create subdirectories
    renders_dir = os.path.join(out_dir, "renders")
    gt_dir = os.path.join(out_dir, "gt")
    pbr_dir = os.path.join(out_dir, "pbr_shaded")
    albedo_dir = os.path.join(out_dir, "albedo")
    roughness_dir = os.path.join(out_dir, "roughness")
    metallic_dir = os.path.join(out_dir, "metallic")
    normal_dir = os.path.join(out_dir, "normal")
    depth_dir = os.path.join(out_dir, "depth")

    for d in [renders_dir, gt_dir, pbr_dir, albedo_dir, roughness_dir, metallic_dir, normal_dir, depth_dir]:
        makedirs(d, exist_ok=True)

    # Load model
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=True)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get cameras based on split
    if split_name == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()

    print(f"\nRendering {len(cameras)} {split_name} views...")

    for idx, view in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        # Render with PBR G-Buffer
        render_pkg = render(view, gaussians, pipeline, background, render_pbr=True)

        # Standard SH rendering
        image = render_pkg["render"]
        save_image(image, os.path.join(renders_dir, f"{view.image_name}.png"))

        # Ground truth
        gt = view.original_image[:3].cuda()
        save_image(gt, os.path.join(gt_dir, f"{view.image_name}.png"))

        # PBR outputs
        if gaussians.use_pbr and 'gbuffer_albedo' in render_pkg:
            gbuffer_albedo = render_pkg['gbuffer_albedo']
            gbuffer_roughness = render_pkg['gbuffer_roughness']
            gbuffer_metallic = render_pkg['gbuffer_metallic']
            gbuffer_normal = render_pkg['rend_normal']
            gbuffer_depth = render_pkg['surf_depth']

            # Save material maps
            save_image(gbuffer_albedo, os.path.join(albedo_dir, f"{view.image_name}.png"))
            save_single_channel(gbuffer_roughness, os.path.join(roughness_dir, f"{view.image_name}.png"))
            save_single_channel(gbuffer_metallic, os.path.join(metallic_dir, f"{view.image_name}.png"))

            # Save normal (convert from [-1, 1] to [0, 1])
            normal_vis = gbuffer_normal * 0.5 + 0.5
            save_image(normal_vis, os.path.join(normal_dir, f"{view.image_name}.png"))

            # Save depth (normalized)
            depth_norm = gbuffer_depth / (gbuffer_depth.max() + 1e-6)
            save_single_channel(depth_norm, os.path.join(depth_dir, f"{view.image_name}.png"), colormap='turbo')

            # PBR shaded image
            shaded = screen_space_pbr_shading(
                gbuffer_albedo,
                gbuffer_roughness,
                gbuffer_metallic,
                gbuffer_normal,
                gbuffer_depth,
                view.camera_center,
                view.world_view_transform,
                env_light=env_light,
            )
            save_image(shaded, os.path.join(pbr_dir, f"{view.image_name}.png"))

    return scene


def compute_metrics(renders_dir, gt_dir):
    """Compute PSNR, SSIM, LPIPS metrics"""
    from utils.loss_utils import ssim
    from utils.image_utils import psnr

    try:
        from lpipsPyTorch import lpips
        has_lpips = True
    except ImportError:
        has_lpips = False
        print("Warning: lpipsPyTorch not available, skipping LPIPS")

    renders = []
    gts = []

    for fname in sorted(os.listdir(renders_dir)):
        if not fname.endswith('.png'):
            continue
        render_img = Image.open(os.path.join(renders_dir, fname))
        gt_img = Image.open(os.path.join(gt_dir, fname))

        render_tensor = torch.from_numpy(np.array(render_img)).float().permute(2, 0, 1)[:3] / 255.0
        gt_tensor = torch.from_numpy(np.array(gt_img)).float().permute(2, 0, 1)[:3] / 255.0

        renders.append(render_tensor.unsqueeze(0).contiguous().cuda())
        gts.append(gt_tensor.unsqueeze(0).contiguous().cuda())

    ssims = []
    psnrs = []
    lpipss = []

    for r, g in zip(renders, gts):
        ssims.append(ssim(r, g).item())
        psnrs.append(psnr(r, g).mean().item())
        if has_lpips:
            lpipss.append(lpips(r, g, net_type='vgg').item())

    metrics = {
        "PSNR": np.mean(psnrs),
        "SSIM": np.mean(ssims),
    }
    if has_lpips:
        metrics["LPIPS"] = np.mean(lpipss)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(description="PBR Rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--env_map", type=str, default=None, help="Path to HDR environment map")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute PSNR/SSIM/LPIPS metrics")

    args = get_combined_args(parser)
    print("Rendering (PBR mode) " + args.model_path)

    # Handle custom args that might not be in config file
    env_map_path = getattr(args, 'env_map', None)
    compute_metrics_flag = getattr(args, 'compute_metrics', False)
    quiet_flag = getattr(args, 'quiet', False)

    safe_state(quiet_flag)

    dataset = model.extract(args)
    iteration = args.iteration
    pipe = pipeline.extract(args)

    # Load environment light
    env_light = EnvironmentLight(env_map_path, resolution=256).cuda()
    print(f"Environment light loaded: {env_map_path if env_map_path else 'default'}")

    # Determine iteration
    if iteration == -1:
        # Find latest iteration
        point_cloud_path = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(point_cloud_path):
            iterations = [int(d.split("_")[-1]) for d in os.listdir(point_cloud_path) if d.startswith("iteration_")]
            if iterations:
                iteration = max(iterations)
                print(f"Using iteration: {iteration}")

    skip_train = getattr(args, 'skip_train', False)
    skip_test = getattr(args, 'skip_test', False)

    # Render train set
    if not skip_train:
        train_dir = os.path.join(args.model_path, 'train', f"ours_{iteration}")
        scene = render_set(dataset, iteration, pipe, env_light,
                          "train", train_dir, "train")

        if compute_metrics_flag:
            print("\nComputing train metrics...")
            metrics = compute_metrics(
                os.path.join(train_dir, "renders"),
                os.path.join(train_dir, "gt")
            )
            print(f"  Train SH:  PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")

            # Also compute for PBR shaded
            pbr_metrics = compute_metrics(
                os.path.join(train_dir, "pbr_shaded"),
                os.path.join(train_dir, "gt")
            )
            print(f"  Train PBR: PSNR={pbr_metrics['PSNR']:.2f}, SSIM={pbr_metrics['SSIM']:.4f}")

    # Render test set
    if not skip_test:
        test_dir = os.path.join(args.model_path, 'test', f"ours_{iteration}")
        scene = render_set(dataset, iteration, pipe, env_light,
                          "test", test_dir, "test")

        if compute_metrics_flag:
            print("\nComputing test metrics...")
            metrics = compute_metrics(
                os.path.join(test_dir, "renders"),
                os.path.join(test_dir, "gt")
            )
            print(f"  Test SH:  PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")

            # Also compute for PBR shaded
            pbr_metrics = compute_metrics(
                os.path.join(test_dir, "pbr_shaded"),
                os.path.join(test_dir, "gt")
            )
            print(f"  Test PBR: PSNR={pbr_metrics['PSNR']:.2f}, SSIM={pbr_metrics['SSIM']:.4f}")

    print("\nRendering complete!")
    print(f"Output saved to: {args.model_path}")
