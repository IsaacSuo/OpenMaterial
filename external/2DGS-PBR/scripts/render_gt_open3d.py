import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
from tqdm import tqdm

# Add project root to sys.path to import scene.colmap_loader
sys.path.append(os.getcwd())

try:
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, qvec2rotmat
except ImportError:
    print("Error: Could not import colmap_loader. Make sure you are running this script from the project root.")
    sys.exit(1)

def load_colmap_cameras(source_path):
    """
    Load camera parameters from COLMAP sparse output.
    """
    sparse_dir = os.path.join(source_path, "sparse", "0")
    if not os.path.exists(sparse_dir):
        # Try finding anywhere inside sparse
        if os.path.exists(os.path.join(source_path, "sparse")):
             subdirs = os.listdir(os.path.join(source_path, "sparse"))
             if len(subdirs) > 0:
                 sparse_dir = os.path.join(source_path, "sparse", subdirs[0])
    
    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"Could not find COLMAP sparse folder in {source_path}")

    # Try binary first, then text
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    
    if os.path.exists(cameras_bin) and os.path.exists(images_bin):
        print(f"Loading COLMAP binary data from {sparse_dir}")
        cameras = read_intrinsics_binary(cameras_bin)
        images = read_extrinsics_binary(images_bin)
    else:
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")
        if os.path.exists(cameras_txt) and os.path.exists(images_txt):
            print(f"Loading COLMAP text data from {sparse_dir}")
            cameras = read_intrinsics_text(cameras_txt)
            images = read_extrinsics_text(images_txt)
        else:
            raise FileNotFoundError("Could not find cameras.bin/.txt or images.bin/.txt")

    return cameras, images

def render_gt(source_path, mesh_path=None, scale_depth=1000.0):
    """
    Render GT Depth and Normal maps using Open3D.
    """
    if mesh_path is None:
        mesh_path = os.path.join(source_path, "points3D.ply")
    
    if not os.path.exists(mesh_path):
        # Try searching for any .ply file if points3D.ply doesn't exist or is invalid
        potential_meshes = [f for f in os.listdir(source_path) if f.endswith('.ply')]
        if len(potential_meshes) > 0:
            mesh_path = os.path.join(source_path, potential_meshes[0])
            print(f"points3D.ply not found, using {mesh_path} instead.")
        else:
            raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Check if mesh has triangles
    if len(mesh.triangles) == 0:
        print("Warning: Mesh has no triangles! Attempting to load as PointCloud (rendering will be sparse/invalid for GT).")
        print("Please ensure your .ply file is a valid Triangle Mesh.")
        # Attempt to proceed but results will likely be blank for standard rasterizer
        # Just return to avoid crash
        return

    # Compute vertex normals if missing
    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()

    # Create output directories
    depth_dir = os.path.join(source_path, "depth_gt")
    normal_dir = os.path.join(source_path, "normal_gt")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # Load COLMAP data
    cameras, images = load_colmap_cameras(source_path)

    # Initialize Visualizer
    # Note: On headless servers, this requires EGL. Open3D 0.19 usually handles this if EGL is present.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1, height=1, visible=False) # Dim doesn't matter yet, will resize per image
    vis.add_geometry(mesh)
    
    # Render Options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) # Background black
    opt.light_on = False # No lighting for normal map painting
    
    # Prepare Normal Color Mesh (clone to avoid modifying original for depth)
    mesh_normal = o3d.geometry.TriangleMesh(mesh)
    # Map normals from [-1, 1] to [0, 1] for RGB
    normals = np.asarray(mesh_normal.vertex_normals)
    colors = (normals + 1.0) * 0.5
    mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)

    print(f"Starting rendering for {len(images)} images...")
    
    sorted_image_keys = sorted(images.keys())
    
    for img_id in tqdm(sorted_image_keys):
        img_data = images[img_id]
        cam_data = cameras[img_data.camera_id]
        
        W, H = cam_data.width, cam_data.height
        
        # Resize window to match camera resolution
        # Note: Open3D headless might not support dynamic resizing perfectly in loop without recreation
        # But capture_depth_float_buffer usually respects window size.
        # Efficient approach: Re-create window only if size changes (usually constant)
        # For simplicity/robustness here, we assume constant or just update view control
        
        # Update intrinsics
        # COLMAP Pinhole: fx, fy, cx, cy
        params = cam_data.params
        if cam_data.model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif cam_data.model == "SIMPLE_PINHOLE":
            fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        elif cam_data.model == "SIMPLE_RADIAL":
            fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        elif cam_data.model == "RADIAL":
             fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        elif cam_data.model == "OPENCV":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            # Fallback for complex models (approximate as PINHOLE)
            # Usually params[0] is f, [1/2] are c
            # This might be inaccurate for Fisheye!
            print(f"Warning: Approx camera model {cam_data.model}")
            fx, fy, cx, cy = params[0], params[0], cam_data.width/2, cam_data.height/2

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        
        # Update Extrinsics
        # COLMAP: World -> Camera (R, T)
        # R is quaternion in qvec
        R = qvec2rotmat(img_data.qvec)
        T = img_data.tvec
        
        # Build 4x4 matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        
        # Convert coordinate system:
        # COLMAP: Right(+X), Down(+Y), Forward(+Z)
        # Open3D: Right(+X), Up(+Y), Back(+Z) (OpenGL style)
        # Transform: Flip Y and Z axis
        # This is equivalent to multiplying by diag(1, -1, -1, 1)
        # But Open3D's convert_from_pinhole_camera_parameters expects the standard W2C matrix 
        # that projects points to camera space. Open3D camera looks down -Z. 
        # COLMAP camera looks down +Z.
        # So we need to flip the Z axis (and Y to match Up/Down convention).
        
        # Correct conversion for Open3D visualizer:
        # T_o3d = T_flip @ T_colmap
        # where T_flip = diag(1, -1, -1, 1)
        convert_mat = np.diag([1, -1, -1, 1])
        extrinsic_o3d = convert_mat @ extrinsic
        
        # 1. Render Normal
        # Use mesh with painted colors
        vis.clear_geometries()
        vis.add_geometry(mesh_normal)
        
        ctr = vis.get_view_control()
        # Note: convert_from_pinhole_camera_parameters sets the EXTRINSIC matrix directly
        # The extrinsic matrix here should be World-to-Camera
        ctr.convert_from_pinhole_camera_parameters(intrinsic, extrinsic_o3d)
        vis.poll_events()
        vis.update_renderer()
        
        normal_rgb = vis.capture_screen_float_buffer(do_render=True)
        normal_rgb = np.asarray(normal_rgb)
        # Open3D captures [0, 1] float RGB
        # Convert back to [-1, 1] if saving as float, or [0, 255] for image
        # Standard: Save as [0, 255] RGB image
        normal_img = (normal_rgb * 255).astype(np.uint8)
        # Open3D is RGB, OpenCV saves BGR
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(normal_dir, f"{img_data.name}"), normal_img)

        # 2. Render Depth
        # Use original mesh
        vis.clear_geometries()
        vis.add_geometry(mesh)
        
        # Reset camera (sometimes needed after clearing)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(intrinsic, extrinsic_o3d)
        vis.poll_events()
        vis.update_renderer()
        
        depth = vis.capture_depth_float_buffer(do_render=True)
        depth = np.asarray(depth)
        
        # Save depth
        # Depth is in mesh units. 
        # Save as 16-bit PNG (mm) or similar
        depth_mm = (depth * scale_depth).astype(np.uint16)
        
        # Ensure path exists (handle subdirectories in image names like 'train/001.jpg')
        save_path_depth = os.path.join(depth_dir, img_data.name)
        save_path_depth = os.path.splitext(save_path_depth)[0] + ".png" # Force png
        os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)
        
        cv2.imwrite(save_path_depth, depth_mm)

    vis.destroy_window()
    print("Done! GT maps generated.")
    print(f"Depth saved to: {depth_dir} (Scale: {scale_depth})")
    print(f"Normal saved to: {normal_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render GT Depth and Normal from Mesh using Open3D")
    parser.add_argument("-s", "--source_path", required=True, help="Path to COLMAP dataset root")
    parser.add_argument("--mesh_name", default=None, help="Specific mesh filename (default points3D.ply)")
    parser.add_argument("--scale", type=float, default=1000.0, help="Depth scale factor for 16-bit PNG (default 1000 for mm)")
    
    args = parser.parse_args()
    
    render_gt(args.source_path, args.mesh_name, args.scale)
