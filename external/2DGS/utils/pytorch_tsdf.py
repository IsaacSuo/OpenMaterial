"""
Pure PyTorch TSDF Fusion Implementation

Replaces Open3D TSDF for better CUDA compatibility.
Uses scikit-image marching cubes for mesh extraction.
"""

import torch
import numpy as np
from tqdm import tqdm


class PyTorchTSDF:
    """
    TSDF (Truncated Signed Distance Function) Volume using PyTorch

    This implementation is compatible with any CUDA version and doesn't
    depend on Open3D's TSDF module.
    """

    def __init__(self, voxel_size, origin, dims, sdf_trunc=None, device='cuda'):
        """
        Initialize TSDF volume

        Args:
            voxel_size: Size of each voxel in meters
            origin: [x, y, z] origin of the volume (min corner)
            dims: [nx, ny, nz] number of voxels in each dimension
            sdf_trunc: SDF truncation distance (default: 5 * voxel_size)
            device: 'cuda' or 'cpu'
        """
        self.voxel_size = voxel_size
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.dims = dims
        self.sdf_trunc = sdf_trunc if sdf_trunc else 5.0 * voxel_size
        self.device = device

        # Initialize TSDF volume and weights
        # tsdf: signed distance values (positive = in front of surface)
        # weight: accumulated weights for weighted average
        # color: RGB color values
        self.tsdf = torch.ones(dims[0], dims[1], dims[2], device=device, dtype=torch.float32)
        self.weight = torch.zeros(dims[0], dims[1], dims[2], device=device, dtype=torch.float32)
        self.color = torch.zeros(dims[0], dims[1], dims[2], 3, device=device, dtype=torch.float32)

        # Pre-compute voxel centers
        self._compute_voxel_coords()

    def _compute_voxel_coords(self):
        """Pre-compute voxel center coordinates"""
        # Create grid of voxel indices
        x = torch.arange(self.dims[0], device=self.device, dtype=torch.float32)
        y = torch.arange(self.dims[1], device=self.device, dtype=torch.float32)
        z = torch.arange(self.dims[2], device=self.device, dtype=torch.float32)

        # Voxel centers in world coordinates
        self.voxel_x = self.origin[0] + (x + 0.5) * self.voxel_size
        self.voxel_y = self.origin[1] + (y + 0.5) * self.voxel_size
        self.voxel_z = self.origin[2] + (z + 0.5) * self.voxel_size

    @torch.no_grad()
    def integrate(self, depth, color, intrinsic, extrinsic, depth_trunc=5.0):
        """
        Integrate a single RGBD frame into the TSDF volume

        Args:
            depth: [H, W] depth image in meters
            color: [H, W, 3] or [3, H, W] RGB image (0-255 or 0-1)
            intrinsic: [3, 3] camera intrinsic matrix
            extrinsic: [4, 4] camera extrinsic matrix (world to camera)
            depth_trunc: Maximum depth to integrate
        """
        H, W = depth.shape[-2:]

        # Ensure tensors are on correct device
        depth = depth.to(self.device).float()
        if depth.dim() == 3:
            depth = depth.squeeze(0)

        # Handle color format
        if isinstance(color, np.ndarray):
            color = torch.from_numpy(color).to(self.device).float()
        else:
            color = color.to(self.device).float()

        if color.dim() == 3 and color.shape[0] == 3:
            color = color.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        if color.max() > 1.0:
            color = color / 255.0

        intrinsic = torch.tensor(intrinsic, device=self.device, dtype=torch.float32) if not isinstance(intrinsic, torch.Tensor) else intrinsic.to(self.device).float()
        extrinsic = torch.tensor(extrinsic, device=self.device, dtype=torch.float32) if not isinstance(extrinsic, torch.Tensor) else extrinsic.to(self.device).float()

        # Camera parameters
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # Process in chunks to avoid OOM
        chunk_size = 128

        for i_start in range(0, self.dims[0], chunk_size):
            i_end = min(i_start + chunk_size, self.dims[0])

            for j_start in range(0, self.dims[1], chunk_size):
                j_end = min(j_start + chunk_size, self.dims[1])

                # Get voxel coordinates for this chunk
                vx = self.voxel_x[i_start:i_end]
                vy = self.voxel_y[j_start:j_end]
                vz = self.voxel_z

                # Create meshgrid for chunk
                gx, gy, gz = torch.meshgrid(vx, vy, vz, indexing='ij')

                # World coordinates [N, 3]
                world_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

                # Transform to camera coordinates
                # world_pts: [N, 3], extrinsic: [4, 4]
                ones = torch.ones(world_pts.shape[0], 1, device=self.device)
                world_pts_h = torch.cat([world_pts, ones], dim=1)  # [N, 4]
                cam_pts = (extrinsic @ world_pts_h.T).T[:, :3]  # [N, 3]

                # Project to image plane
                z_cam = cam_pts[:, 2]

                # Skip points behind camera
                valid_depth = z_cam > 0.01

                u = (fx * cam_pts[:, 0] / z_cam + cx)
                v = (fy * cam_pts[:, 1] / z_cam + cy)

                # Check bounds
                valid_proj = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & valid_depth

                # Get depth values at projected locations (bilinear interpolation)
                u_valid = u[valid_proj].long()
                v_valid = v[valid_proj].long()

                depth_vals = depth[v_valid, u_valid]

                # Compute SDF
                sdf = depth_vals - z_cam[valid_proj]

                # Truncate SDF
                valid_sdf = (sdf > -self.sdf_trunc) & (depth_vals > 0) & (depth_vals < depth_trunc)
                sdf = torch.clamp(sdf / self.sdf_trunc, -1.0, 1.0)

                # Get colors
                colors_valid = color[v_valid, u_valid]

                # Update TSDF volume
                # Map valid indices back to chunk
                chunk_shape = (i_end - i_start, j_end - j_start, self.dims[2])
                flat_indices = torch.arange(world_pts.shape[0], device=self.device)

                # Indices within chunk
                chunk_i = flat_indices // (chunk_shape[1] * chunk_shape[2])
                chunk_j = (flat_indices % (chunk_shape[1] * chunk_shape[2])) // chunk_shape[2]
                chunk_k = flat_indices % chunk_shape[2]

                # Global indices
                global_i = chunk_i + i_start
                global_j = chunk_j + j_start
                global_k = chunk_k

                # Update only valid voxels
                update_mask = valid_proj.clone()
                update_mask[valid_proj] = valid_sdf

                valid_global_i = global_i[update_mask]
                valid_global_j = global_j[update_mask]
                valid_global_k = global_k[update_mask]
                valid_sdf_vals = sdf[valid_sdf]
                valid_colors = colors_valid[valid_sdf]

                # Weighted average update
                old_weight = self.weight[valid_global_i, valid_global_j, valid_global_k]
                old_tsdf = self.tsdf[valid_global_i, valid_global_j, valid_global_k]
                old_color = self.color[valid_global_i, valid_global_j, valid_global_k]

                new_weight = old_weight + 1.0
                self.tsdf[valid_global_i, valid_global_j, valid_global_k] = \
                    (old_weight * old_tsdf + valid_sdf_vals) / new_weight
                self.color[valid_global_i, valid_global_j, valid_global_k] = \
                    (old_weight.unsqueeze(-1) * old_color + valid_colors) / new_weight.unsqueeze(-1)
                self.weight[valid_global_i, valid_global_j, valid_global_k] = new_weight

    def extract_mesh(self, min_weight=1.0):
        """
        Extract triangle mesh using marching cubes

        Args:
            min_weight: Minimum weight threshold for valid voxels

        Returns:
            trimesh.Trimesh object
        """
        from skimage import measure
        import trimesh

        # Get TSDF values, mask out low-weight voxels
        tsdf_np = self.tsdf.cpu().numpy()
        weight_np = self.weight.cpu().numpy()
        color_np = self.color.cpu().numpy()

        # Set unobserved voxels to positive (outside surface)
        tsdf_np[weight_np < min_weight] = 1.0

        # Run marching cubes
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                tsdf_np,
                level=0,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
        except ValueError as e:
            print(f"Marching cubes failed: {e}")
            print("This usually means no surface was found in the TSDF volume")
            # Return empty mesh
            return trimesh.Trimesh()

        # Transform to world coordinates
        verts = verts + self.origin.cpu().numpy()

        # Get vertex colors by interpolation
        # Convert vertex positions to voxel indices
        voxel_coords = (verts - self.origin.cpu().numpy()) / self.voxel_size
        voxel_coords = np.clip(voxel_coords, 0, np.array(self.dims) - 1).astype(int)

        vertex_colors = color_np[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        vertex_colors = (vertex_colors * 255).astype(np.uint8)

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            vertex_colors=vertex_colors
        )

        return mesh

    def get_volume_bounds(self):
        """Get the bounds of the TSDF volume"""
        min_bound = self.origin.cpu().numpy()
        max_bound = min_bound + np.array(self.dims) * self.voxel_size
        return min_bound, max_bound


def create_tsdf_from_cameras(viewpoint_stack, voxel_size=0.004, margin=0.1):
    """
    Create a TSDF volume with bounds computed from camera positions

    Args:
        viewpoint_stack: List of camera viewpoints
        voxel_size: Voxel size in meters
        margin: Extra margin around scene bounds

    Returns:
        PyTorchTSDF instance
    """
    # Compute scene bounds from camera positions
    cam_positions = []
    for vp in viewpoint_stack:
        # Get camera center in world coordinates
        R = vp.R  # [3, 3]
        T = vp.T  # [3]
        # Camera center: C = -R^T @ T
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)
            T = torch.from_numpy(T)
        cam_center = -R.T @ T
        cam_positions.append(cam_center.cpu().numpy())

    cam_positions = np.array(cam_positions)

    # Compute bounds with margin
    min_bound = cam_positions.min(axis=0) - margin
    max_bound = cam_positions.max(axis=0) + margin

    # Compute volume dimensions
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Limit maximum dimensions to avoid OOM
    max_dim = 512
    if dims.max() > max_dim:
        scale = max_dim / dims.max()
        voxel_size = voxel_size / scale
        dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
        print(f"Warning: Volume too large, adjusting voxel_size to {voxel_size:.4f}")

    print(f"TSDF Volume: dims={dims}, voxel_size={voxel_size:.4f}")
    print(f"  Bounds: [{min_bound}] to [{max_bound}]")
    print(f"  Memory: ~{dims[0]*dims[1]*dims[2]*4*2/1e9:.2f} GB")

    return PyTorchTSDF(
        voxel_size=voxel_size,
        origin=min_bound.tolist(),
        dims=dims.tolist(),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def integrate_rgbd_frames(tsdf, viewpoint_stack, rgbmaps, depthmaps, depth_trunc=5.0):
    """
    Integrate multiple RGBD frames into TSDF volume

    Args:
        tsdf: PyTorchTSDF instance
        viewpoint_stack: List of camera viewpoints
        rgbmaps: List of RGB images [C, H, W]
        depthmaps: List of depth images [1, H, W]
        depth_trunc: Maximum depth
    """
    for i, (vp, rgb, depth) in tqdm(enumerate(zip(viewpoint_stack, rgbmaps, depthmaps)),
                                     total=len(viewpoint_stack),
                                     desc="TSDF integration"):
        # Get camera matrices
        # Build intrinsic matrix
        fx = vp.FoVx  # This might be FoV, need to convert
        fy = vp.FoVy
        W, H = vp.image_width, vp.image_height

        # Convert FoV to focal length if needed
        import math
        if fx < 10:  # Likely FoV in radians
            fx = W / (2 * math.tan(fx / 2))
            fy = H / (2 * math.tan(fy / 2))

        intrinsic = np.array([
            [fx, 0, W / 2],
            [0, fy, H / 2],
            [0, 0, 1]
        ])

        # Build extrinsic matrix (world to camera)
        R = vp.R.cpu().numpy() if isinstance(vp.R, torch.Tensor) else vp.R
        T = vp.T.cpu().numpy() if isinstance(vp.T, torch.Tensor) else vp.T

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.T  # Transpose for world-to-camera
        extrinsic[:3, 3] = T

        # Integrate frame
        tsdf.integrate(
            depth=depth,
            color=rgb,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_trunc=depth_trunc
        )

    return tsdf
