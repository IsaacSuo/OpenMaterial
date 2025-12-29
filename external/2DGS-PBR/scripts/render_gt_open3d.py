import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
import json
from tqdm import tqdm
from pathlib import Path

def load_blender_cameras(source_path):
    """
    Load camera parameters from transforms_train.json (Blender/NeRF Synthetic format).
    """
    json_path = os.path.join(source_path, "transforms_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"transforms_train.json not found in {source_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_angle_x = data.get("camera_angle_x")
    frames = data.get("frames", [])
    
    cameras = []
    
    print(f"Loading {len(frames)} cameras from {json_path}")
    
    for frame in frames:
        # File path usually relative to json, e.g. "./train/r_0"
        file_path = frame["file_path"]
        # Remove extension if present, or just ensure basename
        image_name = os.path.basename(file_path).split('.')[0]
        
        # Transform matrix is 4x4 C2W (Camera to World)
        # Coordinate system: OpenGL/Blender (Right=X, Up=Y, Back=Z) -> Camera looks down -Z
        c2w = np.array(frame["transform_matrix"])
        
        # Determine Width/Height
        # Usually not in per-frame data. 
        # We try to read the actual image to get dims.
        full_img_path = os.path.join(source_path, file_path)
        if not full_img_path.endswith(".png") and not full_img_path.endswith(".jpg"):
            full_img_path += ".png" # Assume png
            
        if os.path.exists(full_img_path):
            img = cv2.imread(full_img_path)
            if img is not None:
                H, W = img.shape[:2]
            else:
                # Fallback defaults
                W, H = 800, 800
        else:
            # Fallback
            W, H = 800, 800

        # Calculate intrinsics from FOV
        # focal = 0.5 * W / tan(0.5 * fov)
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        
        cameras.append({
            "name": image_name,
            "c2w": c2w,
            "W": W,
            "H": H,
            "fx": focal,
            "fy": focal,
            "cx": W / 2.0,
            "cy": H / 2.0
        })
        
    return cameras

def find_gt_mesh(source_path, user_mesh_path=None):
    """
    Locate the GT mesh.
    Priority:
    1. Explicitly provided path via -g/--gt_path
    2. OpenMaterial specific structure (datasets/groundtruth_ablation/...)
    3. Local points3d.ply
    """
    # 1. Explicit path
    if user_mesh_path:
        if os.path.exists(user_mesh_path):
            print(f"Using explicitly provided GT mesh: {user_mesh_path}")
            return user_mesh_path
        else:
            print(f"Warning: Provided GT path {user_mesh_path} does not exist. Falling back to auto-discovery.")

    # 2. OpenMaterial Structure Auto-discovery
    try:
        path_obj = Path(source_path).resolve()
        dataset_hash = path_obj.parent.name
        datasets_root = path_obj.parent.parent
        
        gt_ablation_root = datasets_root / "groundtruth_ablation"
        
        if gt_ablation_root.exists():
            gt_hash_dir = gt_ablation_root / dataset_hash
            if gt_hash_dir.exists():
                # Look for clean_<HASH>.ply
                expected_name = f"clean_{dataset_hash}.ply"
                mesh_path = gt_hash_dir / expected_name
                if mesh_path.exists():
                    print(f"Found OpenMaterial GT mesh (auto): {mesh_path}")
                    return str(mesh_path)
                
                # Fallback: any ply
                candidates = list(gt_hash_dir.glob("*.ply"))
                if candidates:
                    print(f"Found GT mesh (fallback name): {candidates[0]}")
                    return str(candidates[0])
    except Exception as e:
        print(f"Auto-discovery failed: {e}")

    # 3. Local fallback
    p3d = os.path.join(source_path, "points3d.ply")
    if os.path.exists(p3d):
        print(f"Using local points3d.ply (Warning: might be sparse): {p3d}")
        return p3d

    raise FileNotFoundError("Could not find GT mesh. Please provide it using -g/--gt_path.")

def render_gt(source_path, mesh_path=None, scale_depth=1000.0):
    
    # 1. Find Mesh
    real_mesh_path = find_gt_mesh(source_path, mesh_path)
    
    print(f"Loading mesh from {real_mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(real_mesh_path)
    if len(mesh.triangles) == 0:
        print("Warning: Mesh has no triangles! Rendering will fail.")
        return

    mesh.compute_vertex_normals()

    # 2. Output dirs
    # We save in the source_path subdirs
    depth_dir = os.path.join(source_path, "depth_gt")
    normal_dir = os.path.join(source_path, "normal_gt")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # 3. Load Cameras
    cameras = load_blender_cameras(source_path)

    # 4. Setup Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1, height=1, visible=False)
    vis.add_geometry(mesh)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.light_on = False

    # Prepare Normal Mesh
    mesh_normal = o3d.geometry.TriangleMesh(mesh)
    # Open3D Normals are usually [-1, 1]. Map to [0, 1] for color.
    # Note: This creates Object Space Normals colored in RGB.
    # If you need View Space Normals, shader is needed. 
    # But usually World Space (Object Space) normals are used for supervision in 2DGS/NeRF.
    # We assume World Space Normals here.
    normals = np.asarray(mesh_normal.vertex_normals)
    colors = (normals + 1.0) * 0.5
    mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)

    print(f"Rendering {len(cameras)} views...")
    
    for cam in tqdm(cameras):
        W, H = cam["W"], cam["H"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        
        # Extrinsic: World-to-Camera
        # JSON gives C2W. Open3D needs W2C (Extrinsic).
        c2w = cam["c2w"]
        w2c = np.linalg.inv(c2w)
        
        # Coordinate System Check:
        # JSON (Blender): +X Right, +Y Up, +Z Back (Camera looks -Z)
        # Open3D: +X Right, +Y Up, +Z Back (Camera looks -Z)
        # They match! No flipping needed.
        # However, Open3D's convert_from_pinhole_camera_parameters expects the extrinsic to be standard W2C.
        
        extrinsic = w2c

        # 1. Render Normal
        vis.clear_geometries()
        vis.add_geometry(mesh_normal)
        ctr = vis.get_view_control()
        # Important: convert_from_pinhole... sets the camera parameters.
        ctr.convert_from_pinhole_camera_parameters(intrinsic, extrinsic)
        vis.poll_events()
        vis.update_renderer()
        
        normal_rgb = vis.capture_screen_float_buffer(do_render=True)
        normal_rgb = np.asarray(normal_rgb)
        
        # Resize if capture didn't match W/H (Open3D headless window size limitations)
        if normal_rgb.shape[0] != H or normal_rgb.shape[1] != W:
            normal_rgb = cv2.resize(normal_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        normal_img = (normal_rgb * 255).astype(np.uint8)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(normal_dir, f"{cam['name']}.png"), normal_img)

        # 2. Render Depth
        vis.clear_geometries()
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(intrinsic, extrinsic)
        vis.poll_events()
        vis.update_renderer()
        
        depth = vis.capture_depth_float_buffer(do_render=True)
        depth = np.asarray(depth)
        
        if depth.shape[0] != H or depth.shape[1] != W:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            
        depth_mm = (depth * scale_depth).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{cam['name']}.png"), depth_mm)

    vis.destroy_window()
    print("GT Generation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-g", "--gt_path", default=None, help="Explicit path to GT mesh (.ply/.obj)")
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()
    
    render_gt(args.source_path, args.gt_path, args.scale)
