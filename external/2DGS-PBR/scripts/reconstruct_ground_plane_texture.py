#!/usr/bin/env python3
"""
Reconstruct a ground-plane texture (e.g., checkerboard floor) from multi-view images.

Key idea:
- An envmap cannot represent finite-depth backgrounds (parallax). A planar surface can.
- Fit a dominant plane from a dense point cloud (PLY), treat it as the ground plane.
- Reproject pixels that likely belong to that plane (via inlier-point projections) and mosaic
  them into a plane atlas (texture) in plane coordinates.

This is designed for the "mask = foreground object" convention:
- We exclude foreground/object pixels (mask==1) from the mosaic.
- We only collect pixels supported by the plane inlier reprojections to avoid painting
  buildings/sky onto the ground.

Outputs (under --out_dir):
- ground_plane.json: plane params + basis + bounds
- ground_texture.png: reconstructed plane texture atlas
- ground_weight.png: coverage/weight visualization
"""

from __future__ import annotations

import sys
from pathlib import Path as _PathForInit
# Add project root to Python path for imports
sys.path.insert(0, str(_PathForInit(__file__).parent.parent.resolve()))

import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import fov2focal


@dataclass
class Plane:
    n: np.ndarray  # (3,)
    d: float
    p0: np.ndarray  # (3,)
    u: np.ndarray  # (3,)
    v: np.ndarray  # (3,)


def _fit_plane_open3d(points: np.ndarray, dist_thresh: float, ransac_n: int, num_iter: int):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=float(dist_thresh),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iter),
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise RuntimeError("Plane normal is degenerate.")
    n = n / nn
    d = float(d) / float(nn)
    return n.astype(np.float32), d, np.asarray(inliers, dtype=np.int64)


def _plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n.astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(ref, n)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u.astype(np.float32), v.astype(np.float32)


def _compute_plane_coords(points: np.ndarray, plane: Plane) -> np.ndarray:
    rel = points - plane.p0[None, :]
    x = rel @ plane.u
    y = rel @ plane.v
    return np.stack([x, y], axis=1)


def _compute_ray_dirs_world_for_pixels(cam, xy: torch.Tensor) -> torch.Tensor:
    """
    Compute world ray directions for selected pixels, matching repo's camera conventions.
    xy: [N,2] with x in [0,W-1], y in [0,H-1]
    """
    device = xy.device
    H, W = int(cam.image_height), int(cam.image_width)
    fovx, fovy = float(cam.FoVx), float(cam.FoVy)
    fx = float(fov2focal(fovx, W))
    fy = float(fov2focal(fovy, H))
    cx = W / 2.0
    cy = H / 2.0

    x = (xy[:, 0] - cx) / fx
    y = (xy[:, 1] - cy) / fy
    z = torch.ones_like(x)
    dirs_c = torch.stack([x, y, z], dim=-1)  # [N,3]
    dirs_c = F.normalize(dirs_c, dim=-1)

    # Camera.world_view_transform is stored transposed; true w2c is world_view_transform.T.
    w2c = cam.world_view_transform.transpose(0, 1)
    c2w_R = w2c[:3, :3].T
    dirs_w = dirs_c @ c2w_R.T
    return F.normalize(dirs_w, dim=-1)


def _project_points_to_image(cam, points_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project world points to pixel coords for a Camera.
    Returns:
      xy: [N,2] float (x,y)
      z:  [N] depth in camera space (positive in front)
    """
    H, W = int(cam.image_height), int(cam.image_width)
    fx = float(fov2focal(float(cam.FoVx), W))
    fy = float(fov2focal(float(cam.FoVy), H))
    cx = W / 2.0
    cy = H / 2.0

    ones = torch.ones((points_w.shape[0], 1), device=points_w.device, dtype=points_w.dtype)
    p_h = torch.cat([points_w, ones], dim=1)  # [N,4]
    w2c = cam.world_view_transform.transpose(0, 1)  # [4,4]
    p_c_h = p_h @ w2c  # row-vector convention
    p_c = p_c_h[:, :3]
    z = p_c[:, 2]
    x = p_c[:, 0] / (z + 1e-8)
    y = p_c[:, 1] / (z + 1e-8)
    px = x * fx + cx
    py = y * fy + cy
    xy = torch.stack([px, py], dim=1)
    return xy, z


def _compute_view_dirs_to_points(cam, points_w: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized view directions from camera center to world points.
    Returns: [N,3] normalized directions
    """
    cam_center = cam.camera_center.view(1, 3).to(points_w.device)
    dirs = points_w - cam_center
    return F.normalize(dirs, dim=-1)


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return (
        F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=radius)
        .squeeze(0)
        .squeeze(0)
    )


def _plane_from_ymin(points: np.ndarray, up_axis: str, offset: float = 0.0) -> Plane:
    """
    Infer a horizontal plane at the minimum coordinate along up_axis.
    Plane equation: n·x + d = 0.
    """
    up_axis = up_axis.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if up_axis not in axis_map:
        raise ValueError(f"--up_axis must be one of x/y/z, got {up_axis}")
    a = axis_map[up_axis]
    h = float(points[:, a].min()) + float(offset)

    n = np.zeros(3, dtype=np.float32)
    n[a] = 1.0
    d = -h

    # Stable axis-aligned basis.
    if up_axis == "y":
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        p0 = np.array([0.0, h, 0.0], dtype=np.float32)
    elif up_axis == "z":
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        p0 = np.array([0.0, 0.0, h], dtype=np.float32)
    else:  # up_axis == "x"
        u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        p0 = np.array([h, 0.0, 0.0], dtype=np.float32)

    return Plane(n=n, d=float(d), p0=p0, u=u, v=v)


@torch.no_grad()
def _sample_image_nn(img_chw: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor sample from a [3,H,W] image at integer pixel indices.
    Returns [N,3].
    """
    return img_chw[:, ys, xs].permute(1, 0).contiguous()


@torch.no_grad()
def _collect_consistent_plane_samples(
    cams: list,
    plane: Plane,
    *,
    per_view_samples: int,
    exclude_mask_threshold: float,
    consistency_views: int,
    consistency_min_agree: int,
    consistency_color_l1: float,
    seed: int,
    min_abs_n_dot_dir: float,
    max_t: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (u,v) plane coords and RGB colors from pixels that are consistent across views.

    Returns:
      uv:  [M,2] float32 plane coordinates
      rgb: [M,3] float32 colors in [0,1]
    """
    device = "cuda"
    rng = np.random.default_rng(int(seed))

    n_t = torch.from_numpy(plane.n).to(device=device, dtype=torch.float32).view(1, 3)
    d_t = torch.tensor(float(plane.d), device=device, dtype=torch.float32)
    p0_t = torch.from_numpy(plane.p0).to(device=device, dtype=torch.float32).view(1, 3)
    u_t = torch.from_numpy(plane.u).to(device=device, dtype=torch.float32).view(3, 1)
    v_t = torch.from_numpy(plane.v).to(device=device, dtype=torch.float32).view(3, 1)

    uv_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []

    num_cams = len(cams)
    if num_cams < 2:
        raise RuntimeError("Need at least 2 views for reprojection consistency.")

    for cam_idx, cam in enumerate(cams):
        H, W = int(cam.image_height), int(cam.image_width)
        img = cam.original_image[:3].to(device)  # [3,H,W]

        fg = cam.gt_alpha_mask.to(device) if cam.gt_alpha_mask is not None else None
        if fg is not None:
            fg = (fg.squeeze(0) > float(exclude_mask_threshold))
            bg = ~fg
            print(f"[Cam {cam_idx}] Mask stats: FG pixels={fg.sum().item()}, BG pixels={bg.sum().item()}")
        else:
            bg = torch.ones((H, W), device=device, dtype=torch.bool)
            print(f"[Cam {cam_idx}] No mask, using full image")

        ys, xs = torch.nonzero(bg, as_tuple=True)
        if ys.numel() == 0:
            print(f"[Cam {cam_idx}] WARNING: No background pixels found!")
            continue

        if per_view_samples > 0 and ys.numel() > per_view_samples:
            sel = torch.randperm(ys.numel(), device=device)[:per_view_samples]
            ys = ys[sel]
            xs = xs[sel]

        xy = torch.stack([xs.float(), ys.float()], dim=1)  # [N,2]
        dirs_w = _compute_ray_dirs_world_for_pixels(cam, xy)  # [N,3]
        o = cam.camera_center.view(1, 3).to(device)  # [1,3]

        denom = (dirs_w * n_t).sum(dim=1)  # [N]
        if float(min_abs_n_dot_dir) > 0:
            ang_ok = torch.abs(denom) >= float(min_abs_n_dot_dir)
            denom = denom[ang_ok]
            dirs_w = dirs_w[ang_ok]
            xs = xs[ang_ok]
            ys = ys[ang_ok]
            xy = xy[ang_ok]
            if denom.numel() == 0:
                continue
        numer = (o * n_t).sum(dim=1).squeeze(0) + d_t  # scalar
        t = -numer / (denom + 1e-8)
        valid_t = t > 0
        if max_t is not None and float(max_t) > 0:
            valid_t = valid_t & (t < float(max_t))
        if valid_t.sum() == 0:
            continue

        t = t[valid_t].view(-1, 1)
        dirs_w = dirs_w[valid_t]
        xs = xs[valid_t]
        ys = ys[valid_t]

        print(f"[Cam {cam_idx}] Ray-plane intersections: {t.shape[0]} valid hits")

        p = o + t * dirs_w  # [N,3]
        rel = p - p0_t
        pu = (rel @ u_t).squeeze(1)
        pv = (rel @ v_t).squeeze(1)

        print(f"[Cam {cam_idx}] Plane coords range: u=[{pu.min().item():.2f}, {pu.max().item():.2f}], v=[{pv.min().item():.2f}, {pv.max().item():.2f}]")

        src_rgb = _sample_image_nn(img, xs.long(), ys.long())  # [N,3]

        k = min(int(consistency_views), num_cams - 1)
        if k <= 0:
            continue
        tgt_indices = rng.choice([i for i in range(num_cams) if i != cam_idx], size=k, replace=False)
        agree = torch.zeros((p.shape[0],), device=device, dtype=torch.int32)

        for j in tgt_indices:
            cam_j = cams[int(j)]
            Hj, Wj = int(cam_j.image_height), int(cam_j.image_width)
            img_j = cam_j.original_image[:3].to(device)
            fg_j = cam_j.gt_alpha_mask.to(device) if cam_j.gt_alpha_mask is not None else None
            if fg_j is not None:
                fg_j = (fg_j.squeeze(0) > float(exclude_mask_threshold))

            xy_j, z_j = _project_points_to_image(cam_j, p)
            valid = (
                (z_j > 1e-4)
                & (xy_j[:, 0] >= 0)
                & (xy_j[:, 0] < Wj - 1)
                & (xy_j[:, 1] >= 0)
                & (xy_j[:, 1] < Hj - 1)
            )
            if valid.sum() == 0:
                continue

            xi = xy_j[valid, 0].round().long().clamp(0, Wj - 1)
            yi = xy_j[valid, 1].round().long().clamp(0, Hj - 1)
            tgt_rgb = _sample_image_nn(img_j, xi, yi)  # [M,3]

            ok = torch.ones((tgt_rgb.shape[0],), device=device, dtype=torch.bool)
            if fg_j is not None:
                ok = ok & (~fg_j[yi, xi])

            diff = torch.abs(tgt_rgb - src_rgb[valid]).mean(dim=1)
            ok = ok & (diff <= float(consistency_color_l1))

            agree_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            agree[agree_idx[ok]] += 1

        keep = agree >= int(consistency_min_agree)
        print(f"[Cam {cam_idx}] Consistency check: {keep.sum().item()}/{len(keep)} samples passed (min_agree={consistency_min_agree})")

        if keep.sum() == 0:
            continue

        pu_k = pu[keep].detach().cpu().numpy().astype(np.float32)
        pv_k = pv[keep].detach().cpu().numpy().astype(np.float32)
        rgb_k = src_rgb[keep].detach().cpu().numpy().astype(np.float32)

        uv_all.append(np.stack([pu_k, pv_k], axis=1))
        rgb_all.append(rgb_k)

        print(f"[Cam {cam_idx}] Total collected: {len(pu_k)} samples")

        if (cam_idx + 1) % 10 == 0:
            print(f"[{cam_idx+1}/{num_cams}] collected consistent samples")

    if not uv_all:
        raise RuntimeError("No consistent ground samples found. Try increasing --per_view_samples or loosening thresholds.")
    return np.concatenate(uv_all, axis=0), np.concatenate(rgb_all, axis=0)


def main():
    parser = ArgumentParser(description="Reconstruct a ground-plane texture atlas from multi-view images.")
    model = ModelParams(parser)

    parser.add_argument("--ply", type=str, required=True, help="Point cloud PLY (for plane fitting or ymin inference)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Plane fitting
    parser.add_argument("--plane_mode", type=str, default="fit", choices=["fit", "ymin"])
    parser.add_argument("--up_axis", type=str, default="y", help="Used for plane_mode=ymin; one of x/y/z")
    parser.add_argument("--ymin_offset", type=float, default=0.0, help="Used for plane_mode=ymin; add offset to ymin height")
    parser.add_argument("--plane_dist_thresh", type=float, default=0.01)
    parser.add_argument("--plane_ransac_n", type=int, default=3)
    parser.add_argument("--plane_num_iter", type=int, default=2000)

    # Mosaic
    parser.add_argument("--tex_res", type=int, default=2048, help="Target texture height; width is set by plane aspect")
    parser.add_argument("--plane_margin", type=float, default=0.05, help="Extra margin (ratio) added to plane bounds")
    parser.add_argument("--per_view_samples", type=int, default=200000, help="Max pixels sampled per view (0 = all supported)")
    parser.add_argument("--support_dilate", type=int, default=3, help="(fit mode) Dilate projected inlier support mask by this radius")
    parser.add_argument("--exclude_mask_threshold", type=float, default=0.5, help="Foreground mask threshold (mask>thr excluded)")
    parser.add_argument("--use_test_views", action="store_true", help="Also use test views for mosaic")

    # Multi-view consistency (ymin mode)
    parser.add_argument("--consistency_views", type=int, default=4, help="Number of other views to check per sample")
    parser.add_argument("--consistency_min_agree", type=int, default=2, help="Min agreeing views to accept a sample")
    parser.add_argument("--consistency_color_l1", type=float, default=0.12, help="Mean absolute RGB threshold for agreement (LDR in [0,1])")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min_abs_n_dot_dir",
        type=float,
        default=0.15,
        help="Reject rays nearly parallel to the plane: require |n·dir| >= this value (helps avoid huge t/uv outliers).",
    )
    parser.add_argument(
        "--max_t",
        type=float,
        default=None,
        help="Optional max ray-plane intersection distance (in world units). If set, rejects very far intersections.",
    )
    parser.add_argument(
        "--uv_quantile",
        type=float,
        default=0.01,
        help="Use quantiles [q, 1-q] to compute uv bounds in ymin mode (robust to outliers).",
    )

    args = parser.parse_args()
    dataset = model.extract(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cameras without initializing gaussians.
    src = dataset.source_path
    if os.path.exists(os.path.join(src, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](src, dataset.images, dataset.eval)
    elif os.path.exists(os.path.join(src, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](src, dataset.white_background, dataset.eval)
    else:
        raise RuntimeError(f"Unrecognized dataset type at: {src}")

    train_cams = cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)
    test_cams = cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset) if args.use_test_views else []
    cams = list(train_cams) + list(test_cams)
    if not cams:
        raise RuntimeError("No cameras found.")

    # Load PLY points (fast path via plyfile isn't used here; rely on repo loader if desired).
    # We implement a tiny PLY reader via numpy for xyz fields.
    from plyfile import PlyData

    ply = PlyData.read(args.ply)
    verts = ply["vertex"]
    pts = np.stack([np.asarray(verts["x"]), np.asarray(verts["y"]), np.asarray(verts["z"])], axis=1).astype(np.float32)

    if args.plane_mode == "fit":
        n, d, inliers = _fit_plane_open3d(pts, args.plane_dist_thresh, args.plane_ransac_n, args.plane_num_iter)
        inlier_pts = pts[inliers]
        p0 = inlier_pts.mean(axis=0).astype(np.float32)
        u, v = _plane_basis_from_normal(n)
        plane = Plane(n=n, d=float(d), p0=p0, u=u, v=v)

        uv = _compute_plane_coords(inlier_pts, plane)
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv_s = None
        rgb_s = None
    else:
        plane = _plane_from_ymin(pts, up_axis=str(args.up_axis), offset=float(args.ymin_offset))

        uv_s, rgb_s = _collect_consistent_plane_samples(
            cams,
            plane,
            per_view_samples=int(args.per_view_samples),
            exclude_mask_threshold=float(args.exclude_mask_threshold),
            consistency_views=int(args.consistency_views),
            consistency_min_agree=int(args.consistency_min_agree),
            consistency_color_l1=float(args.consistency_color_l1),
            seed=int(args.seed),
            min_abs_n_dot_dir=float(args.min_abs_n_dot_dir),
            max_t=float(args.max_t) if args.max_t is not None else None,
        )
        q = float(args.uv_quantile)
        q = max(0.0, min(0.49, q))
        uv_min = np.quantile(uv_s, q, axis=0)
        uv_max = np.quantile(uv_s, 1.0 - q, axis=0)

    margin = float(args.plane_margin)
    span = uv_max - uv_min
    uv_min = uv_min - margin * span
    uv_max = uv_max + margin * span
    span = uv_max - uv_min

    tex_h = int(args.tex_res)
    aspect = float(span[0] / max(span[1], 1e-9))
    tex_w = int(round(tex_h * aspect))
    tex_w = max(64, tex_w)

    # Accumulators on CPU (float32)
    tex_sum = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    tex_wsum = np.zeros((tex_h, tex_w, 1), dtype=np.float32)

    if args.plane_mode == "ymin":
        print(f"\n=== Texture accumulation (ymin mode) ===")
        print(f"Collected samples: {len(uv_s)}")
        print(f"UV bounds for texture: min={uv_min}, max={uv_max}")
        print(f"Span: {span}")
        print(f"Texture size: {tex_w} x {tex_h}")

        uu = (uv_s[:, 0] - float(uv_min[0])) / float(span[0] + 1e-12)
        vv = (uv_s[:, 1] - float(uv_min[1])) / float(span[1] + 1e-12)
        inside = (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
        print(f"Samples inside texture bounds: {inside.sum()}/{len(inside)}")

        uu = uu[inside]
        vv = vv[inside]
        rgb = rgb_s[inside]

        if len(uu) == 0:
            print("WARNING: No samples inside texture bounds!")
        else:
            tx = uu * (tex_w - 1)
            ty = (1.0 - vv) * (tex_h - 1)
            x0 = np.floor(tx).astype(np.int64)
            y0 = np.floor(ty).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, tex_w - 1)
            y1 = np.clip(y0 + 1, 0, tex_h - 1)
            wx = np.clip(tx - x0.astype(np.float32), 0.0, 1.0)
            wy = np.clip(ty - y0.astype(np.float32), 0.0, 1.0)
            w00 = ((1 - wx) * (1 - wy))[:, None]
            w10 = (wx * (1 - wy))[:, None]
            w01 = ((1 - wx) * wy)[:, None]
            w11 = (wx * wy)[:, None]
            for (xi, yi, wi) in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
                np.add.at(tex_sum, (yi, xi), rgb * wi)
                np.add.at(tex_wsum, (yi, xi), wi)
            print(f"Accumulated {len(uu)} samples into texture")
    else:
        inlier_pts_t = torch.from_numpy(inlier_pts).to(device="cuda", dtype=torch.float32)
        for cam_idx, cam in enumerate(cams):
            H, W = int(cam.image_height), int(cam.image_width)
            img = cam.original_image[:3].to("cuda")  # [3,H,W]

            with torch.no_grad():
                xy, z = _project_points_to_image(cam, inlier_pts_t)
                valid = (z > 1e-4) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
                xyv = xy[valid]
                if xyv.numel() == 0:
                    continue
                xi = xyv[:, 0].round().long().clamp(0, W - 1)
                yi = xyv[:, 1].round().long().clamp(0, H - 1)
                support = torch.zeros((H, W), device="cuda", dtype=torch.float32)
                support[yi, xi] = 1.0
                support = _dilate_mask(support, int(args.support_dilate))

                fg = cam.gt_alpha_mask.to("cuda") if cam.gt_alpha_mask is not None else None
                if fg is not None:
                    fg = (fg.squeeze(0) > float(args.exclude_mask_threshold)).to(torch.float32)  # [H,W]
                    support = support * (1.0 - fg)

                ys, xs = torch.nonzero(support > 0.5, as_tuple=True)
                if ys.numel() == 0:
                    continue

                if int(args.per_view_samples) > 0 and ys.numel() > int(args.per_view_samples):
                    sel = torch.randperm(ys.numel(), device="cuda")[: int(args.per_view_samples)]
                    ys = ys[sel]
                    xs = xs[sel]

                xy_pix = torch.stack([xs.float(), ys.float()], dim=1)  # [N,2]
                dirs_w = _compute_ray_dirs_world_for_pixels(cam, xy_pix)  # [N,3]
                o = cam.camera_center.view(1, 3).to("cuda")  # [1,3]

                n_t = torch.from_numpy(plane.n).to(device="cuda", dtype=torch.float32).view(1, 3)
                d_t = torch.tensor(float(plane.d), device="cuda", dtype=torch.float32)
                denom = (dirs_w * n_t).sum(dim=1)  # [N]
                numer = (o * n_t).sum(dim=1).squeeze(0) + d_t  # scalar
                t = -numer / (denom + 1e-8)
                valid_t = t > 0
                if valid_t.sum() == 0:
                    continue

                t = t[valid_t].view(-1, 1)
                dirs_w = dirs_w[valid_t]
                xs = xs[valid_t]
                ys = ys[valid_t]

                p = o + t * dirs_w  # [N,3]
                p0_t = torch.from_numpy(plane.p0).to(device="cuda", dtype=torch.float32).view(1, 3)
                u_t = torch.from_numpy(plane.u).to(device="cuda", dtype=torch.float32).view(3, 1)
                v_t = torch.from_numpy(plane.v).to(device="cuda", dtype=torch.float32).view(3, 1)
                rel = p - p0_t
                pu = (rel @ u_t).squeeze(1)
                pv = (rel @ v_t).squeeze(1)

                uu = (pu - float(uv_min[0])) / float(span[0] + 1e-12)
                vv = (pv - float(uv_min[1])) / float(span[1] + 1e-12)
                inside = (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
                if inside.sum() == 0:
                    continue

                uu = uu[inside]
                vv = vv[inside]
                xs = xs[inside]
                ys = ys[inside]

                colors = img[:, ys, xs].permute(1, 0).contiguous()  # [N,3]

                tx = uu * (tex_w - 1)
                ty = (1.0 - vv) * (tex_h - 1)
                x0 = torch.floor(tx).long()
                y0 = torch.floor(ty).long()
                x1 = (x0 + 1).clamp(0, tex_w - 1)
                y1 = (y0 + 1).clamp(0, tex_h - 1)
                wx = (tx - x0.float()).clamp(0.0, 1.0)
                wy = (ty - y0.float()).clamp(0.0, 1.0)

                w00 = (1 - wx) * (1 - wy)
                w10 = wx * (1 - wy)
                w01 = (1 - wx) * wy
                w11 = wx * wy

                x0c = x0.detach().cpu().numpy()
                y0c = y0.detach().cpu().numpy()
                x1c = x1.detach().cpu().numpy()
                y1c = y1.detach().cpu().numpy()
                c = colors.detach().cpu().numpy()
                w00c = w00.detach().cpu().numpy()[:, None]
                w10c = w10.detach().cpu().numpy()[:, None]
                w01c = w01.detach().cpu().numpy()[:, None]
                w11c = w11.detach().cpu().numpy()[:, None]

                for (xi, yi, wi) in ((x0c, y0c, w00c), (x1c, y0c, w10c), (x0c, y1c, w01c), (x1c, y1c, w11c)):
                    np.add.at(tex_sum, (yi, xi), c * wi)
                    np.add.at(tex_wsum, (yi, xi), wi)

            if (cam_idx + 1) % 10 == 0:
                print(f"[{cam_idx+1}/{len(cams)}] processed views")

    tex = tex_sum / np.clip(tex_wsum, 1e-6, None)
    tex = np.clip(tex, 0.0, 1.0)
    weight = tex_wsum.squeeze(-1)
    w_vis = weight / (weight.max() + 1e-8)

    Image.fromarray((tex * 255.0).astype(np.uint8)).save(out_dir / "ground_texture.png")
    Image.fromarray((w_vis * 255.0).astype(np.uint8)).save(out_dir / "ground_weight.png")

    meta = {
        "plane": {"n": plane.n.tolist(), "d": float(plane.d)},
        "basis": {"p0": plane.p0.tolist(), "u": plane.u.tolist(), "v": plane.v.tolist()},
        "uv_bounds": {"min": uv_min.tolist(), "max": uv_max.tolist()},
        "tex_size": {"h": int(tex_h), "w": int(tex_w)},
        "notes": "Texture coords: u along basis u, v along basis v; v axis is flipped in the PNG (top=+v_max).",
    }
    (out_dir / "ground_plane.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved: {out_dir/'ground_texture.png'}")
    print(f"Saved: {out_dir/'ground_plane.json'}")


if __name__ == "__main__":
    main()
