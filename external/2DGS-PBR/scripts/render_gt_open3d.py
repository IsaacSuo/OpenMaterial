import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
import json
from tqdm import tqdm
from pathlib import Path

# Force EGL usage for headless rendering
os.environ["EGL_PLATFORM"] = "surfaceless"

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
    Locate the GT mesh using the specific OpenMaterial directory structure.
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

def render_gt(source_path, mesh_path=None, scale_depth=1000.0, debug=False):
    
    # 1. Find Mesh
    real_mesh_path = find_gt_mesh(source_path, mesh_path)
    
    print(f"Loading mesh from {real_mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(real_mesh_path)
    if len(mesh.triangles) == 0:
        print("Warning: Mesh has no triangles! Rendering will fail.")
        return

    # Check/Fix normals
    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()

    # Debug: print mesh stats
    vertices = np.asarray(mesh.vertices)
    print(f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Mesh bounds: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
          f"Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], "
          f"Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    # 2. Output dirs
    depth_dir = os.path.join(source_path, "depth_gt")
    normal_dir = os.path.join(source_path, "normal_gt")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # 3. Load Cameras
    cameras = load_blender_cameras(source_path)

    # 4. Prepare Geometry
    # We create a single mesh with Vertex Colors = Normals
    # This allows us to render the normal map by just rendering the mesh with an Unlit shader
    mesh_normal = o3d.geometry.TriangleMesh(mesh)
    normals = np.asarray(mesh_normal.vertex_normals)

    # Note: The normals are in world space. When rendered, they will be transformed
    # by the view matrix. Since we want camera-space normals (for consistency with
    # 2DGS which outputs camera-space normals), we store world-space normals here
    # and they get transformed during rendering.

    # Map [-1, 1] -> [0, 1]
    colors = (normals + 1.0) * 0.5
    mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)

    if debug:
        cameras = cameras[:1]  # Only render first view in debug mode
        print(f"[DEBUG] Rendering 1 view (debug mode)")
    else:
        print(f"Rendering {len(cameras)} views using OffscreenRenderer...")

    # 5. Initialize Renderer loop
    renderer = None
    current_w, current_h = 0, 0

    for cam in tqdm(cameras, disable=debug):
        W, H = cam["W"], cam["H"]
        
        # Initialize or Re-initialize renderer if dimensions change
        if renderer is None or W != current_w or H != current_h:
            # Need to create a new renderer if size changes
            renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
            current_w, current_h = W, H
            
            # Setup scene geometry (only needs to be done once per renderer instance)
            # Use 'defaultUnlit' to render vertex colors (which are normals) directly without shading
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit" 
            
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("obj", mesh_normal, mat)
            
            # Set background to Black
            renderer.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))

        # Setup Camera
        # Intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, cam["fx"], cam["fy"], cam["cx"], cam["cy"])

        # Extrinsic (World-to-Camera)
        c2w = cam["c2w"]

        # Use OpenGL coordinate system directly (Y up, Z back)
        # This matches dataset_readers.py which no longer flips coordinates
        w2c = np.linalg.inv(c2w)

        if debug:
            cam_pos = c2w[:3, 3]
            print(f"[DEBUG] Camera '{cam['name']}': pos={cam_pos}, W={W}, H={H}")
            print(f"[DEBUG] c2w:\n{c2w}")
            print(f"[DEBUG] w2c:\n{w2c}")

        # Setup camera in renderer
        # Note: OffscreenRenderer.setup_camera takes (intrinsic, extrinsic_matrix)
        renderer.setup_camera(intrinsic, w2c)
        
        # 1. Render Normal Map (RGB)
        # Since shader is Unlit and vertex colors are normals, the RGB output is the normal map
        img_o3d = renderer.render_to_image()
        img_np = np.asarray(img_o3d)

        if debug:
            print(f"[DEBUG] Normal image: shape={img_np.shape}, range=[{img_np.min()}, {img_np.max()}], nonzero={np.count_nonzero(img_np)}")

        # Convert RGB to BGR for OpenCV
        normal_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(normal_dir, f"{cam['name']}.png"), normal_bgr)
        
        # 2. Render Depth Map
        # render_to_depth_image returns float depth in view space
        depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
        depth_np = np.asarray(depth_o3d)

        if debug:
            print(f"[DEBUG] Raw depth: min={depth_np.min():.4f}, max={depth_np.max():.4f}, "
                  f"inf_count={(np.isinf(depth_np)).sum()}, non_inf_max={depth_np[~np.isinf(depth_np)].max() if (~np.isinf(depth_np)).any() else 'N/A'}")

        # Check for inf/nan
        depth_np[np.isinf(depth_np)] = 0
        depth_np[np.isnan(depth_np)] = 0

        if debug:
            print(f"[DEBUG] After cleanup: min={depth_np.min():.4f}, max={depth_np.max():.4f}, nonzero={np.count_nonzero(depth_np)}")

        # Scale and save as 16-bit PNG
        depth_mm = (depth_np * scale_depth).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{cam['name']}.png"), depth_mm)

        if debug:
            print(f"[DEBUG] Saved depth: {depth_mm.min()} ~ {depth_mm.max()}")

    print("GT Generation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-g", "--gt_path", default=None, help="Explicit path to GT mesh (.ply/.obj)")
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--debug", action="store_true", help="Debug mode: render only first view with verbose output")
    args = parser.parse_args()

    render_gt(args.source_path, args.gt_path, args.scale, debug=args.debug)
