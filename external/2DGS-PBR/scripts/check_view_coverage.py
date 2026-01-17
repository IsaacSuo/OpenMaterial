#!/usr/bin/env python3
"""
Check view-direction coverage between train and test splits.

Designed for Blender/NeRF-synthetic style datasets that provide:
  - transforms_train.json
  - transforms_test.json

This script estimates whether test view directions (or sampled camera rays)
are "covered" by train directions by computing nearest-neighbor angular distance
on the unit sphere (max dot-product).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _load_transforms(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_repo_axis_fix(c2w: np.ndarray) -> np.ndarray:
    """
    Match external/2DGS-PBR/scene/dataset_readers.py:readCamerasFromTransforms:
      c2w[:3, 1:3] *= -1
    """
    c2w = c2w.copy()
    c2w[:3, 1:3] *= -1
    return c2w


def _camera_forward_world(c2w: np.ndarray) -> np.ndarray:
    """
    In this repo's pinhole ray convention, camera-space forward is +Z.
    So the center ray direction in world is c2w_R @ [0,0,1].
    """
    R = c2w[:3, :3]
    f = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _normalize(f)


def _fovx_to_fovy(fovx: float, width: int, height: int) -> float:
    # focal = pixels / (2*tan(fov/2))
    fx = width / (2.0 * math.tan(fovx * 0.5))
    fovy = 2.0 * math.atan(height / (2.0 * fx))
    return float(fovy)


def _sample_camera_rays_world(
    c2w: np.ndarray,
    fovx: float,
    width: int,
    height: int,
    grid: int,
) -> np.ndarray:
    """
    Sample a grid x grid set of rays across the image, in world space.
    Returns [grid*grid, 3] unit vectors.
    """
    assert grid >= 1
    fovy = _fovx_to_fovy(fovx, width=width, height=height)

    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    fx = width / (2.0 * tan_fovx)
    fy = height / (2.0 * tan_fovy)
    cx = width / 2.0
    cy = height / 2.0

    if grid == 1:
        xs = np.array([cx], dtype=np.float32)
        ys = np.array([cy], dtype=np.float32)
    else:
        xs = np.linspace(0, width - 1, grid, dtype=np.float32)
        ys = np.linspace(0, height - 1, grid, dtype=np.float32)

    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    dirs_c = np.stack(
        [
            (xx - cx) / fx,
            (yy - cy) / fy,
            np.ones_like(xx, dtype=np.float32),
        ],
        axis=-1,
    ).reshape(-1, 3)
    dirs_c = _normalize(dirs_c)

    R = c2w[:3, :3]
    dirs_w = dirs_c @ R.T
    return _normalize(dirs_w)


@dataclass(frozen=True)
class SampledDirs:
    dirs: np.ndarray  # [N,3]
    frame_ids: np.ndarray  # [N]
    file_paths: List[str]  # per-frame, indexable by frame_id


def _collect_dirs(
    transforms: dict,
    dataset_root: str,
    mode: str,
    grid: int,
) -> SampledDirs:
    fovx = float(transforms["camera_angle_x"])
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms json.")

    # Determine image size once (assume constant across frames).
    width = height = None
    for fr in frames:
        rel = fr.get("file_path")
        if not rel:
            continue
        img_path = os.path.join(dataset_root, rel)
        if os.path.exists(img_path):
            try:
                from PIL import Image

                with Image.open(img_path) as im:
                    width, height = im.size
                break
            except Exception:
                pass
    if width is None or height is None:
        raise FileNotFoundError(
            "Failed to determine image resolution: could not open any frame image under dataset_root."
        )

    all_dirs: List[np.ndarray] = []
    all_frame_ids: List[np.ndarray] = []
    file_paths: List[str] = []

    for idx, fr in enumerate(frames):
        rel = fr.get("file_path")
        if not rel:
            raise ValueError(f"Frame {idx} missing file_path.")
        file_paths.append(rel)

        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        c2w = _apply_repo_axis_fix(c2w)

        if mode == "center":
            d = _camera_forward_world(c2w)[None, :]
        elif mode == "grid":
            d = _sample_camera_rays_world(c2w, fovx=fovx, width=width, height=height, grid=grid)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_dirs.append(d)
        all_frame_ids.append(np.full((d.shape[0],), idx, dtype=np.int32))

    dirs = np.concatenate(all_dirs, axis=0)
    frame_ids = np.concatenate(all_frame_ids, axis=0)
    return SampledDirs(dirs=dirs.astype(np.float32), frame_ids=frame_ids, file_paths=file_paths)


def _max_dot_nn(train_dirs: np.ndarray, test_dirs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """
    For each test direction, compute max dot with all train directions.
    Returns [N_test] of max dot values.
    """
    train_dirs = _normalize(train_dirs).astype(np.float32)
    test_dirs = _normalize(test_dirs).astype(np.float32)

    out = np.empty((test_dirs.shape[0],), dtype=np.float32)
    train_T = train_dirs.T  # [3, M]
    for i in range(0, test_dirs.shape[0], chunk):
        j = min(i + chunk, test_dirs.shape[0])
        dots = test_dirs[i:j] @ train_T  # [chunk, M]
        out[i:j] = dots.max(axis=1)
    return out


def _deg_from_dot(dot: np.ndarray) -> np.ndarray:
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _summarize_angles(name: str, angles_deg: np.ndarray) -> None:
    pct = [50, 75, 90, 95, 99, 100]
    vals = np.percentile(angles_deg, pct)
    print(f"{name}:")
    for p, v in zip(pct, vals):
        print(f"  p{p:>3}: {v:8.3f} deg")


def _report_worst_frames(
    test_samples: SampledDirs,
    angles_deg: np.ndarray,
    topk: int,
) -> None:
    # Aggregate per-frame: use the worst (max) sampled-ray angle for that frame.
    n_frames = len(test_samples.file_paths)
    worst = np.zeros((n_frames,), dtype=np.float32)
    for frame_id in range(n_frames):
        mask = test_samples.frame_ids == frame_id
        if not np.any(mask):
            continue
        worst[frame_id] = float(angles_deg[mask].max())

    order = np.argsort(-worst)[:topk]
    print(f"\nWorst {len(order)} test frames by max angular gap:")
    for idx in order:
        print(f"  {worst[idx]:8.3f} deg  frame={idx:04d}  path={test_samples.file_paths[idx]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check train/test view-direction coverage.")
    ap.add_argument(
        "--dataset_root",
        "-s",
        required=True,
        help="Dataset root containing transforms_train.json and transforms_test.json.",
    )
    ap.add_argument(
        "--mode",
        choices=["center", "grid"],
        default="center",
        help="Use only the center ray ('center') or sample a grid of rays per frame ('grid').",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=5,
        help="Grid size per frame when --mode grid (default: 5 => 25 rays per frame).",
    )
    ap.add_argument(
        "--threshold_deg",
        type=float,
        default=10.0,
        help="Coverage threshold: test rays with NN angle <= threshold are counted as covered.",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Report worst K test frames by maximum angular gap.",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="Chunk size for dot-product computation (memory/speed tradeoff).",
    )
    args = ap.parse_args()

    train_json = os.path.join(args.dataset_root, "transforms_train.json")
    test_json = os.path.join(args.dataset_root, "transforms_test.json")
    if not os.path.exists(train_json) or not os.path.exists(test_json):
        raise FileNotFoundError(
            f"Expected transforms_train.json and transforms_test.json under {args.dataset_root}"
        )

    train_tf = _load_transforms(train_json)
    test_tf = _load_transforms(test_json)

    train = _collect_dirs(train_tf, dataset_root=args.dataset_root, mode=args.mode, grid=args.grid)
    test = _collect_dirs(test_tf, dataset_root=args.dataset_root, mode=args.mode, grid=args.grid)

    print(f"Mode: {args.mode}" + (f" (grid={args.grid})" if args.mode == "grid" else ""))
    print(f"Train frames: {len(train.file_paths)}  samples: {train.dirs.shape[0]}")
    print(f"Test  frames: {len(test.file_paths)}  samples: {test.dirs.shape[0]}")

    max_dot = _max_dot_nn(train.dirs, test.dirs, chunk=args.chunk)
    angles = _deg_from_dot(max_dot)

    _summarize_angles("Test->Train NN angle", angles)

    covered = (angles <= float(args.threshold_deg)).mean() * 100.0
    print(f"\nCoverage @ {args.threshold_deg:.2f} deg: {covered:.2f}% of test samples")

    _report_worst_frames(test, angles, topk=args.topk)


if __name__ == "__main__":
    main()

