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
from utils.pbr_utils import (
    EnvironmentLight,
    GroundPlane,
    screen_space_pbr_shading,
    compute_ray_directions_world_from_fov,
    tonemap_reinhard,
    linear_to_srgb,
)
from utils.light_probe import LightProbe, LightProbeConfig

import numpy as np
from PIL import Image

def _search_for_max_iteration_in_dir(root: str):
    """
    Find max iteration_* under a directory (e.g., unfixed_point_cloud/iteration_XXXX).
    Returns None if not found.
    """
    if not os.path.isdir(root):
        return None
    max_it = None
    for name in os.listdir(root):
        if not name.startswith("iteration_"):
            continue
        try:
            it = int(name.split("iteration_")[-1])
        except Exception:
            continue
        if max_it is None or it > max_it:
            max_it = it
    return max_it


def _search_for_latest_ckpt(model_path: str, prefix: str) -> str | None:
    """
    Find the latest checkpoint under model_path with pattern: <prefix>_<iter>.pth
    Returns full path or None.
    """
    if not os.path.isdir(model_path):
        return None
    best_it = None
    best_path = None
    for name in os.listdir(model_path):
        if not (name.startswith(prefix + "_") and name.endswith(".pth")):
            continue
        try:
            it = int(name[len(prefix) + 1 :].split(".pth")[0])
        except Exception:
            continue
        if best_it is None or it > best_it:
            best_it = it
            best_path = os.path.join(model_path, name)
    return best_path


def save_image(tensor, path):
    """Save a [C, H, W] tensor as image"""
    img = linear_to_srgb(tonemap_reinhard(tensor)).detach().cpu().clamp(0, 1)
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


def _compute_background(ray_dirs, camera_center, env_light, ground_plane=None):
    """
    Compute background color (ground + sky composite).

    Args:
        ray_dirs: [H, W, 3] ray directions in world space
        camera_center: [3] camera position
        env_light: EnvironmentLight module
        ground_plane: Optional GroundPlane module

    Returns:
        bg: [3, H, W] background color
    """
    sky_color = env_light.sample(ray_dirs).permute(2, 0, 1)

    if ground_plane is None:
        return sky_color

    ground_color, ground_mask = ground_plane.sample(ray_dirs, camera_center)
    ground_color = ground_color.permute(2, 0, 1)
    ground_mask = ground_mask.unsqueeze(0).float()

    return ground_color * ground_mask + sky_color * (1.0 - ground_mask)


def render_set(dataset, iteration, pipeline, env_light, light_probe, views, out_dir, split_name, ground_plane=None):
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

    object_render_mode = str(getattr(dataset, "object_render_mode", "pbr")).lower().strip() if hasattr(dataset, "object_render_mode") else "pbr"
    # Load object model
    gaussians = GaussianModel(dataset.sh_degree, use_pbr=(object_render_mode == "pbr"))
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Optional: load unfixed/background Gaussians (SH-only)
    unfixed_gaussians = None
    unfixed_root = os.path.join(scene.model_path, "unfixed_point_cloud")
    unfixed_it = iteration
    if unfixed_it == -1:
        unfixed_it = _search_for_max_iteration_in_dir(unfixed_root)
    if unfixed_it is not None:
        unfixed_ply = os.path.join(unfixed_root, f"iteration_{unfixed_it}", "point_cloud.ply")
        if os.path.exists(unfixed_ply):
            unfixed_gaussians = GaussianModel(dataset.sh_degree, use_pbr=False)
            unfixed_gaussians.load_ply(unfixed_ply)
            print(f"[Unfixed] Loaded background Gaussians: {unfixed_ply}")

    # Use black background so buffers are premultiplied cleanly; skybox is composited explicitly.
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Get cameras based on split
    if split_name == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()

    print(f"\nRendering {len(cameras)} {split_name} views...")

    for idx, view in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        # Render object (PBR or SH)
        if object_render_mode == "pbr":
            render_pkg = render(view, gaussians, pipeline, background, render_pbr=True)
        else:
            render_pkg = render(view, gaussians, pipeline, background, render_pbr=False)

        # Skybox for composite outputs
        ray_dirs = compute_ray_directions_world_from_fov(
            image_height=view.image_height,
            image_width=view.image_width,
            fovx=view.FoVx,
            fovy=view.FoVy,
            world_view_transform=view.world_view_transform,
            device="cuda",
        )
        # Background: either unfixed SH Gaussians + sky, or ground_plane + sky, or sky only.
        sky = env_light.sample(ray_dirs).permute(2, 0, 1)
        if unfixed_gaussians is not None:
            bg_pkg = render(view, unfixed_gaussians, pipeline, background, render_pbr=False)
            alpha_bg = bg_pkg["rend_alpha"]
            bg_env = bg_pkg["render"] + sky * (1.0 - alpha_bg)
        else:
            bg_env = _compute_background(ray_dirs, view.camera_center, env_light, ground_plane)
        alpha_map = render_pkg["rend_alpha"]

        # Standard SH rendering (always available); in PBR mode this is just for reference.
        image = render_pkg["render"] + bg_env * (1.0 - alpha_map)
        save_image(image, os.path.join(renders_dir, f"{view.image_name}.png"))

        # Ground truth
        gt = view.original_image[:3].cuda()
        save_image(gt, os.path.join(gt_dir, f"{view.image_name}.png"))

        # PBR outputs
        if (object_render_mode == "pbr") and gaussians.use_pbr and 'gbuffer_albedo' in render_pkg:
            denom = alpha_map + 1e-6

            # Unpremultiply to get physically meaningful material maps.
            gbuffer_albedo = torch.clamp(render_pkg['gbuffer_albedo'] / denom, 0.0, 1.0)
            gbuffer_roughness = torch.clamp(render_pkg['gbuffer_roughness'] / denom, 0.1, 0.999)
            gbuffer_metallic = torch.clamp(render_pkg['gbuffer_metallic'] / denom, 0.0, 1.0)
            gbuffer_normal = render_pkg['rend_normal'] / denom
            gbuffer_depth = render_pkg['surf_depth']

            # Save material maps
            save_image(gbuffer_albedo, os.path.join(albedo_dir, f"{view.image_name}.png"))
            save_single_channel(gbuffer_roughness, os.path.join(roughness_dir, f"{view.image_name}.png"))
            save_single_channel(gbuffer_metallic, os.path.join(metallic_dir, f"{view.image_name}.png"))

            # Save normal (normalize, then map [-1, 1] -> [0, 1])
            nrm = gbuffer_normal / (gbuffer_normal.norm(dim=0, keepdim=True) + 1e-6)
            normal_vis = nrm * 0.5 + 0.5
            save_image(normal_vis, os.path.join(normal_dir, f"{view.image_name}.png"))

            # Save depth (normalized)
            depth_norm = gbuffer_depth / (gbuffer_depth.max() + 1e-6)
            save_single_channel(depth_norm, os.path.join(depth_dir, f"{view.image_name}.png"), colormap='turbo')

            # PBR shaded image (object only)
            shaded_obj = screen_space_pbr_shading(
                gbuffer_albedo,
                gbuffer_roughness,
                gbuffer_metallic,
                gbuffer_normal,
                gbuffer_depth,
                view.camera_center,
                view.world_view_transform,
                # If a probe is provided, object shading must NOT call env_map.
                env_light=None if (light_probe is not None) else env_light,
                light_probe=light_probe,
                ray_dirs_world=ray_dirs,
                clamp_output=False,
            )
            shaded = shaded_obj * alpha_map + bg_env * (1.0 - alpha_map)
            save_image(shaded, os.path.join(pbr_dir, f"{view.image_name}.png"))
        elif object_render_mode == "sh":
            # Baseline: mirror SH composite into pbr_shaded for convenience/metrics.
            save_image(image, os.path.join(pbr_dir, f"{view.image_name}.png"))

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

        render_tensor = torch.from_numpy(np.array(render_img)).float().permute(2, 0, 1)[:3].contiguous() / 255.0
        gt_tensor = torch.from_numpy(np.array(gt_img)).float().permute(2, 0, 1)[:3].contiguous() / 255.0

        renders.append(render_tensor.unsqueeze(0).cuda())
        gts.append(gt_tensor.unsqueeze(0).cuda())

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
    parser.add_argument(
        "--object_render_mode",
        type=str,
        default="pbr",
        choices=["pbr", "sh"],
        help="Object rendering mode: 'pbr' or 'sh' (baseline).",
    )
    parser.add_argument(
        "--ground_plane_json",
        type=str,
        default=None,
        help="Path to ground_plane.json (for finite-depth ground background)",
    )
    parser.add_argument(
        "--ground_texture",
        type=str,
        default=None,
        help="Path to ground_texture.png. If not specified, looks in same dir as ground_plane_json.",
    )
    parser.add_argument(
        "--light_model",
        type=str,
        default="envmap",
        choices=["envmap", "probe"],
        help="Lighting model for PBR object shading: 'envmap' (far-field) or 'probe' (position+direction light probe network).",
    )
    parser.add_argument("--probe_pth", type=str, default=None, help="Optional light probe checkpoint to load.")
    parser.add_argument("--probe_backend", type=str, default="tcnn", choices=["tcnn", "mlp"], help="Probe backend.")
    parser.add_argument("--probe_dir_encoding", type=str, default="sh", choices=["sh", "fourier"], help="Direction encoding for probe.")
    parser.add_argument("--probe_sh_degree", type=int, default=4, help="SH degree for direction encoding (tcnn backend).")
    parser.add_argument("--probe_fourier_n_frequencies", type=int, default=6, help="Fourier frequencies for direction encoding.")
    parser.add_argument("--probe_n_levels", type=int, default=16, help="HashGrid levels (tcnn backend).")
    parser.add_argument("--probe_n_features_per_level", type=int, default=2, help="HashGrid features per level (tcnn backend).")
    parser.add_argument("--probe_log2_hashmap_size", type=int, default=19, help="HashGrid log2 hashmap size (tcnn backend).")
    parser.add_argument("--probe_base_resolution", type=int, default=16, help="HashGrid base resolution (tcnn backend).")
    parser.add_argument("--probe_per_level_scale", type=float, default=1.5, help="HashGrid per-level scale (tcnn backend).")
    parser.add_argument("--probe_hidden_dim", type=int, default=64, help="Probe decoder hidden width.")
    parser.add_argument("--probe_n_hidden_layers", type=int, default=2, help="Probe decoder hidden layers.")
    parser.add_argument(
        "--probe_output_activation",
        type=str,
        default="softplus",
        choices=["softplus", "exp", "none"],
        help="Output activation for probe radiance (HDR).",
    )
    parser.add_argument("--probe_output_softplus_beta", type=float, default=1.0, help="Softplus beta for probe output.")
    parser.add_argument("--probe_aabb_min", type=float, nargs=3, default=None, help="Probe AABB min (3 floats).")
    parser.add_argument("--probe_aabb_max", type=float, nargs=3, default=None, help="Probe AABB max (3 floats).")

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
    setattr(dataset, "object_render_mode", getattr(args, "object_render_mode", "pbr"))

    # Determine iteration first
    if iteration == -1:
        # Find latest iteration
        point_cloud_path = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(point_cloud_path):
            iterations = [int(d.split("_")[-1]) for d in os.listdir(point_cloud_path) if d.startswith("iteration_")]
            if iterations:
                iteration = max(iterations)
                print(f"Using iteration: {iteration}")

    # Load environment light (try to load trained one first)
    env_light = EnvironmentLight(env_map_path, resolution=256).cuda()
    env_light_loaded = False

    # Try to load trained environment light
    env_light_path = os.path.join(args.model_path, f"env_light_{iteration}.pth")
    if os.path.exists(env_light_path):
        env_light.load_state_dict(torch.load(env_light_path))
        print(f"Loaded trained environment light from: {env_light_path}")
        env_light_loaded = True
    else:
        # Try to find any saved env_light
        if os.path.exists(args.model_path):
            for f in sorted(os.listdir(args.model_path), reverse=True):
                if f.startswith("env_light_") and f.endswith(".pth"):
                    env_light_path = os.path.join(args.model_path, f)
                    env_light.load_state_dict(torch.load(env_light_path))
                    print(f"Loaded trained environment light from: {env_light_path}")
                    env_light_loaded = True
                    break

    if not env_light_loaded:
        print(f"Using default environment light (no trained env_light found)")

    # Optional: load light probe for object shading.
    light_model = str(getattr(args, "light_model", "envmap")).lower().strip()
    light_probe = None
    if light_model == "probe":
        aabb_min = tuple(float(x) for x in args.probe_aabb_min) if getattr(args, "probe_aabb_min", None) is not None else (-1.0, -1.0, -1.0)
        aabb_max = tuple(float(x) for x in args.probe_aabb_max) if getattr(args, "probe_aabb_max", None) is not None else (1.0, 1.0, 1.0)
        probe_cfg = LightProbeConfig(
            backend=str(getattr(args, "probe_backend", "tcnn")),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            n_levels=int(getattr(args, "probe_n_levels", 16) or 16),
            n_features_per_level=int(getattr(args, "probe_n_features_per_level", 2) or 2),
            log2_hashmap_size=int(getattr(args, "probe_log2_hashmap_size", 19) or 19),
            base_resolution=int(getattr(args, "probe_base_resolution", 16) or 16),
            per_level_scale=float(getattr(args, "probe_per_level_scale", 1.5) or 1.5),
            dir_encoding=str(getattr(args, "probe_dir_encoding", "sh")),
            sh_degree=int(getattr(args, "probe_sh_degree", 4) or 4),
            fourier_n_frequencies=int(getattr(args, "probe_fourier_n_frequencies", 6) or 6),
            hidden_dim=int(getattr(args, "probe_hidden_dim", 64) or 64),
            n_hidden_layers=int(getattr(args, "probe_n_hidden_layers", 2) or 2),
            output_activation=str(getattr(args, "probe_output_activation", "softplus")),
            output_softplus_beta=float(getattr(args, "probe_output_softplus_beta", 1.0) or 1.0),
        )
        light_probe = LightProbe(probe_cfg).cuda()
        probe_ckpt = getattr(args, "probe_pth", None)
        if probe_ckpt is None:
            # Prefer the current iteration, otherwise load the latest.
            p = os.path.join(args.model_path, f"light_probe_{iteration}.pth")
            if os.path.exists(p):
                probe_ckpt = p
            else:
                probe_ckpt = _search_for_latest_ckpt(args.model_path, "light_probe")
        if probe_ckpt is not None and os.path.exists(str(probe_ckpt)):
            light_probe.load_state_dict(torch.load(str(probe_ckpt), map_location="cpu"))
            light_probe = light_probe.cuda()
            print(f"Loaded light probe from: {probe_ckpt}")
        else:
            print("Warning: --light_model=probe but no probe checkpoint found; using randomly initialized probe.")

    # Load ground plane (optional, for finite-depth ground backgrounds)
    ground_plane = None
    ground_plane_json = getattr(args, 'ground_plane_json', None)
    if ground_plane_json and os.path.exists(ground_plane_json):
        tex_path = getattr(args, 'ground_texture', None)
        if tex_path is None:
            tex_path = os.path.join(os.path.dirname(ground_plane_json), "ground_texture.png")
        if os.path.exists(tex_path):
            ground_plane = GroundPlane(json_path=ground_plane_json, texture_path=tex_path).cuda()
            print(f"Loaded ground plane from: {ground_plane_json}")
        else:
            print(f"Warning: Ground texture not found at {tex_path}, skipping ground plane")

    skip_train = getattr(args, 'skip_train', False)
    skip_test = getattr(args, 'skip_test', False)

    # Render train set
    if not skip_train:
        train_dir = os.path.join(args.model_path, 'train', f"ours_{iteration}")
        scene = render_set(dataset, iteration, pipe, env_light, light_probe,
                          "train", train_dir, "train", ground_plane=ground_plane)

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
        scene = render_set(dataset, iteration, pipe, env_light, light_probe,
                          "test", test_dir, "test", ground_plane=ground_plane)

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
    # dataset.object_render_mode is already set before rendering.
