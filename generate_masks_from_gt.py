#!/usr/bin/env python3
"""
Generate mask images from ground truth mesh for scenes missing masks.
Renders GT mesh from each camera viewpoint to create silhouette masks.
"""

import numpy as np
import os
import json
import math
from pathlib import Path
from glob import glob
import argparse

import mitsuba as mi
mi.set_variant('scalar_rgb')
from mitsuba import ScalarTransform4f as T


def get_camera_intrinsic(width, height, fov_x, fov_y):
    """Convert FOV to focal length"""
    fx = width / 2.0 / math.tan(fov_x / 180 * math.pi / 2.0)
    fy = height / 2.0 / math.tan(fov_y / 180 * math.pi / 2.0)
    return fx, fy


def matrix_to_mitsuba_transform(matrix):
    """Convert 4x4 numpy matrix to Mitsuba transform"""
    return T(matrix.tolist())


def render_mask(gt_mesh_path, transform_matrix, width, height, fov_x):
    """
    Render GT mesh from a camera viewpoint to get silhouette mask.

    Args:
        gt_mesh_path: Path to GT mesh (.ply)
        transform_matrix: 4x4 camera-to-world transform (OpenGL convention)
        width, height: Image dimensions
        fov_x: Horizontal field of view in degrees

    Returns:
        mask: Binary mask as numpy array (H, W), True where object is visible
    """
    # Convert transform matrix
    c2w = np.array(transform_matrix)

    # Mitsuba uses look_at, we need to extract position and target
    # Position is the translation part
    position = c2w[:3, 3]

    # Forward direction (negative Z in OpenGL convention)
    forward = -c2w[:3, 2]
    target = position + forward

    # Up direction (Y in OpenGL convention)
    up = c2w[:3, 1]

    # Create scene with GT mesh
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 2,
        },
        'sensor': {
            'type': 'perspective',
            'fov': fov_x,
            'to_world': T.look_at(
                origin=position.tolist(),
                target=target.tolist(),
                up=up.tolist()
            ),
            'film': {
                'type': 'hdrfilm',
                'width': width,
                'height': height,
                'pixel_format': 'rgb',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 1,  # We only need silhouette, 1 sample is enough
            },
        },
        'emitter': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [1.0, 1.0, 1.0],
            },
        },
        'shape': {
            'type': 'ply',
            'filename': str(gt_mesh_path),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [1.0, 1.0, 1.0],  # White material
                },
            },
        },
    }

    # Render
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)

    # Convert to numpy
    image_np = np.array(image)

    # Create binary mask: any pixel with value > 0 is part of the object
    # Sum across channels and check if > threshold
    mask = np.sum(image_np, axis=2) > 0.01

    return mask


def generate_masks_for_scene(gt_mesh_path, transforms_path, output_mask_dir,
                             width=1600, height=1200, fov_x=37.8492):
    """
    Generate masks for all frames in a scene.

    Args:
        gt_mesh_path: Path to ground truth mesh
        transforms_path: Path to transforms_train.json
        output_mask_dir: Directory to save generated masks
    """
    # Load transforms
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    # Create output directory
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    frames = transforms['frames']
    print(f"  Generating {len(frames)} masks...")

    for i, frame in enumerate(frames):
        transform_matrix = frame['transform_matrix']

        # Generate mask
        mask = render_mask(gt_mesh_path, transform_matrix, width, height, fov_x)

        # Convert to uint8 (0 or 255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Save mask - match naming convention from existing masks
        # Get the image filename from frame if available, otherwise use index
        if 'file_path' in frame:
            # Extract filename without extension
            img_name = Path(frame['file_path']).stem
            mask_filename = f"{img_name}.png"
        else:
            mask_filename = f"{i:04d}.png"

        mask_path = output_mask_dir / mask_filename

        # Save using PIL or cv2
        from PIL import Image
        mask_img = Image.fromarray(mask_uint8, mode='L')
        # Convert to RGB (3 channels) to match existing mask format
        mask_rgb = Image.merge('RGB', [mask_img, mask_img, mask_img])
        mask_rgb.save(mask_path)

        if (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{len(frames)} masks")

    print(f"  Saved {len(frames)} masks to {output_mask_dir}")


def find_scenes_without_masks(dataset_dir):
    """Find all scenes that are missing mask directories"""
    dataset_dir = Path(dataset_dir)
    missing_scenes = []

    for obj_dir in dataset_dir.iterdir():
        if not obj_dir.is_dir() or obj_dir.name.startswith('groundtruth'):
            continue

        for scene_dir in obj_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            mask_dir = scene_dir / "train" / "mask"
            transforms_path = scene_dir / "transforms_train.json"

            # Check if masks are missing but transforms exist
            if transforms_path.exists():
                if not mask_dir.exists() or len(list(mask_dir.glob("*.png"))) == 0:
                    missing_scenes.append({
                        'object_name': obj_dir.name,
                        'scene_name': scene_dir.name,
                        'scene_dir': scene_dir,
                        'transforms_path': transforms_path,
                        'mask_dir': mask_dir,
                    })

    return missing_scenes


def main():
    parser = argparse.ArgumentParser(description='Generate masks from GT mesh')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Path to ground truth meshes directory')
    parser.add_argument('--object_name', type=str, default=None,
                        help='Only process specific object (optional)')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='Only process specific scene (optional)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only show what would be done')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    gt_dir = Path(args.gt_dir)

    print("="*60)
    print("Generate Masks from Ground Truth Mesh")
    print("="*60)

    # Find scenes without masks
    missing_scenes = find_scenes_without_masks(dataset_dir)

    # Filter if specific object/scene requested
    if args.object_name:
        missing_scenes = [s for s in missing_scenes if s['object_name'] == args.object_name]
    if args.scene_name:
        missing_scenes = [s for s in missing_scenes if s['scene_name'] == args.scene_name]

    print(f"\nFound {len(missing_scenes)} scenes without masks:")
    for scene in missing_scenes:
        print(f"  - {scene['object_name']}/{scene['scene_name']}")

    if args.dry_run:
        print("\n[Dry run] No masks generated.")
        return

    if len(missing_scenes) == 0:
        print("\nNo scenes to process.")
        return

    print(f"\n{'='*60}")
    print("Generating masks...")
    print("="*60)

    for i, scene in enumerate(missing_scenes):
        obj_name = scene['object_name']
        scene_name = scene['scene_name']

        print(f"\n[{i+1}/{len(missing_scenes)}] {obj_name}/{scene_name}")

        # Find GT mesh
        gt_mesh_candidates = list((gt_dir / obj_name).glob("*.ply"))
        if not gt_mesh_candidates:
            print(f"  ERROR: No GT mesh found in {gt_dir / obj_name}")
            continue

        gt_mesh_path = gt_mesh_candidates[0]
        print(f"  GT mesh: {gt_mesh_path.name}")

        # Generate masks
        try:
            generate_masks_for_scene(
                gt_mesh_path=gt_mesh_path,
                transforms_path=scene['transforms_path'],
                output_mask_dir=scene['mask_dir'],
            )
            print(f"  SUCCESS")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
