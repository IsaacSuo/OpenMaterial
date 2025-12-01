#!/usr/bin/env python3
"""
Convert OpenMaterial dataset to NeuS2 format
Usage: python tools/convert_openmaterial.py --input datasets/openmaterial/OBJECT_ID/SCENE_NAME --output neus2_data/OBJECT_ID/SCENE_NAME
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import shutil


def convert_openmaterial_to_neus2(input_dir, output_dir, split='train'):
    """Convert OpenMaterial format to NeuS2 format"""

    # Read OpenMaterial transforms
    transform_file = os.path.join(input_dir, f'transforms_{split}.json')
    if not os.path.exists(transform_file):
        print(f"Warning: {transform_file} not found, skipping {split} split")
        return False

    with open(transform_file, 'r') as f:
        meta = json.load(f)

    # Get image dimensions
    W = int(meta.get('w', 1600))
    H = int(meta.get('h', 1200))

    # Create output directory structure
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)

    # Prepare NeuS2 format transforms
    neus2_transforms = {
        "from_na": True,  # Use NeuS2's coordinate system
        "w": W,
        "h": H,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [0.5, 0.5, 0.5],
        "frames": []
    }

    # Process each frame
    for idx, frame in enumerate(meta['frames']):
        # Get camera parameters
        c2w = np.array(frame['transform_matrix'])  # 4x4 camera-to-world matrix

        # OpenMaterial uses fl_x, fl_y for focal lengths
        fl_x = meta.get('fl_x', meta.get('camera_angle_x', None))
        fl_y = meta.get('fl_y', meta.get('camera_angle_y', None))

        if fl_x is None or fl_y is None:
            # Fallback: calculate from camera_angle_x
            if 'camera_angle_x' in meta:
                fl_x = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])
                fl_y = fl_x  # Assume square pixels
            else:
                raise ValueError("Cannot determine focal length")

        # Build intrinsic matrix
        cx = W / 2.0
        cy = H / 2.0
        intrinsic_matrix = [
            [fl_x, 0.0, cx, 0.0],
            [0.0, fl_y, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]

        # Get image and mask paths
        img_path = os.path.join(input_dir, frame['file_path'])

        # Determine mask path
        file_path_parts = frame['file_path'].split('/')
        if len(file_path_parts) >= 2:
            mask_path = os.path.join(input_dir, file_path_parts[0], 'mask', file_path_parts[-1])
        else:
            mask_path = None

        # Output image filename
        output_img_name = f"{idx:06d}.png"
        output_img_path = os.path.join(output_images_dir, output_img_name)

        # Load and merge RGB + mask as RGBA
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')

            # Load mask if available
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # Ensure mask same size as image
                if mask.size != img.size:
                    mask = mask.resize(img.size, Image.BICUBIC)
                # Create RGBA image
                img_rgba = Image.new('RGBA', img.size)
                img_rgba.paste(img, (0, 0))
                img_rgba.putalpha(mask)
                img_rgba.save(output_img_path)
            else:
                # No mask, use fully opaque alpha
                img_rgba = Image.new('RGBA', img.size)
                img_rgba.paste(img, (0, 0))
                alpha = Image.new('L', img.size, 255)
                img_rgba.putalpha(alpha)
                img_rgba.save(output_img_path)
        else:
            print(f"Warning: Image {img_path} not found, skipping frame {idx}")
            continue

        # Add frame to transforms
        neus2_transforms['frames'].append({
            "file_path": f"images/{output_img_name}",
            "transform_matrix": c2w.tolist(),
            "intrinsic_matrix": intrinsic_matrix
        })

    # Save NeuS2 format transforms
    output_transform_file = os.path.join(output_dir, f'transforms_{split}.json')
    with open(output_transform_file, 'w') as f:
        json.dump(neus2_transforms, f, indent=2)

    print(f"Converted {len(neus2_transforms['frames'])} frames for {split} split")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert OpenMaterial dataset to NeuS2 format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing OpenMaterial data (e.g., datasets/openmaterial/OBJECT_ID/SCENE_NAME)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for NeuS2 format data')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'],
                        help='Splits to convert (default: train test)')

    args = parser.parse_args()

    print(f"Converting OpenMaterial data from {args.input} to {args.output}")
    print(f"Splits: {args.splits}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Convert each split
    for split in args.splits:
        print(f"\n=== Converting {split} split ===")
        success = convert_openmaterial_to_neus2(args.input, args.output, split)
        if not success:
            print(f"Failed to convert {split} split")

    print(f"\nConversion complete! Output saved to {args.output}")
    print(f"You can now train with: python scripts/run.py --scene {args.output}/transforms_train.json")


if __name__ == '__main__':
    main()
