#!/usr/bin/env python3
"""
Sample Mesh to Dense PLY with Normals

Usage:
    python scripts/sample_mesh_to_ply.py --input_mesh <path_to_mesh> --output_ply <path_to_ply> --num_points 1000000

Dependencies:
    pip install open3d numpy
"""

import open3d as o3d
import numpy as np
import argparse
import os
import sys

def sample_mesh(input_path, output_path, num_points=1000000):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"Loading mesh from: {input_path}")
    # Read mesh
    try:
        mesh = o3d.io.read_triangle_mesh(input_path)
    except Exception as e:
        print(f"Error reading mesh: {e}")
        sys.exit(1)

    if len(mesh.triangles) == 0:
        print("Error: Loaded mesh has no triangles.")
        sys.exit(1)

    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Triangles: {len(mesh.triangles)}")

    # Ensure mesh has normals before sampling
    if not mesh.has_triangle_normals():
        print("  - Computing triangle normals...")
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        print("  - Computing vertex normals...")
        mesh.compute_vertex_normals()

    print(f"Sampling {num_points} points (Poisson Disk)...")
    # Poisson Disk Sampling ensures more even distribution than Uniform
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)

    # Check if normals were sampled correctly
    if not pcd.has_normals():
        print("Warning: Sampled PCD has no normals. Estimating from neighbors...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print(f"Saving dense PLY to: {output_path}")
    
    # Save options: include normals, maybe colors if they exist
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False, print_progress=True)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a mesh to a dense point cloud with normals for Static 2DGS training.")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to input mesh (.obj, .glb, .ply, .stl)")
    parser.add_argument("--output_ply", type=str, required=True, help="Path to output .ply file")
    parser.add_argument("--num_points", type=int, default=1_000_000, help="Number of points to sample (default: 1,000,000)")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    out_dir = os.path.dirname(args.output_ply)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    sample_mesh(args.input_mesh, args.output_ply, args.num_points)
