"""
PGSR (Planar-based Gaussian Splatting) Method Wrapper
"""

from pathlib import Path
from typing import Dict, Any
from ..base_method import BaseMethod


class PGSRMethod(BaseMethod):
    """Wrapper for PGSR method"""

    def __init__(self, repo_path: str = "external/PGSR"):
        super().__init__(
            method_name="pgsr",
            repo_path=repo_path,
            conda_env="pgsr"
        )

    def setup(self) -> bool:
        """Setup PGSR environment"""
        if not self.check_environment():
            print(f"Creating conda environment: {self.conda_env}")
            result = self.run_command(
                f"conda create -n {self.conda_env} python=3.8 -y",
                use_conda=False
            )
            if result.returncode != 0:
                return False

        # Install PyTorch (check if already installed)
        print("Checking PyTorch...")
        check_torch = self.run_command(
            "python -c \"import torch; print(torch.__version__)\" 2>/dev/null"
        )
        if check_torch.returncode == 0 and "2.3.1" in check_torch.stdout:
            print(f"✓ PyTorch 2.3.1 already installed, skipping download")
        else:
            print("Installing PyTorch...")
            result = self.run_command(
                "pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
            if result.returncode != 0:
                return False

        # Install dependencies (including those in requirements.txt)
        print("Checking dependencies...")
        check_deps = self.run_command(
            "python -c \"import open3d; import plyfile; import lpips; import trimesh\" 2>/dev/null"
        )
        if check_deps.returncode == 0:
            print("✓ Dependencies already installed")
        else:
            print("Installing dependencies...")
            result = self.run_command(
                "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
            if result.returncode != 0:
                print(f"Failed to install base dependencies: {result.stderr}")
                return False

        # Install PyTorch3D (required for PGSR mesh processing)
        print("Checking PyTorch3D...")
        check_pytorch3d = self.run_command(
            "python -c \"import pytorch3d; print(pytorch3d.__version__)\" 2>/dev/null"
        )
        if check_pytorch3d.returncode == 0:
            print(f"✓ PyTorch3D already installed")
        else:
            print("Installing PyTorch3D...")
            result = self.run_command(
                "pip install pytorch3d -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
            if result.returncode != 0:
                print(f"⚠ PyPI version failed, trying from source...")
                result = self.run_command(
                    'pip install "git+https://github.com/facebookresearch/pytorch3d.git"'
                )
                if result.returncode != 0:
                    print(f"Failed to install PyTorch3D: {result.stderr}")
                    return False

        # Build CUDA extensions (check if already built)
        print("Checking CUDA extensions...")
        check_extensions = self.run_command(
            "python -c \"import diff_plane_rasterization; import simple_knn\" 2>/dev/null"
        )
        if check_extensions.returncode == 0:
            print("✓ CUDA extensions already built")
        else:
            print("Building CUDA extensions...")
            result = self.run_command("pip install submodules/diff-plane-rasterization")
            if result.returncode != 0:
                print(f"Failed to build diff-plane-rasterization: {result.stderr}")
                return False

            result = self.run_command("pip install submodules/simple-knn")
            if result.returncode != 0:
                print(f"Failed to build simple-knn: {result.stderr}")
                return False

        print("✓ PGSR setup complete")
        return True

    def convert_data(self, input_path: str, output_path: str) -> bool:
        """PGSR uses transforms.json format directly, create symlink to avoid data duplication"""
        import os
        from pathlib import Path

        output_path_obj = Path(output_path)
        input_path_obj = Path(input_path)

        # Create parent directory
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing symlink/directory if it exists
        if output_path_obj.exists() or output_path_obj.is_symlink():
            if output_path_obj.is_symlink():
                output_path_obj.unlink()
            else:
                import shutil
                shutil.rmtree(output_path_obj)

        # Create symlink to input data
        os.symlink(input_path_obj.absolute(), output_path_obj)

        return True

    def train(self, data_path: str, output_path: str, **kwargs) -> bool:
        """Train PGSR"""
        from pathlib import Path

        config = self.get_default_config()
        config.update(kwargs)

        iterations = config.get('iterations', 30000)
        densify_abs_grad_threshold = config.get('densify_abs_grad_threshold', 0.0008)
        max_abs_split_points = config.get('max_abs_split_points', 50000)
        opacity_cull_threshold = config.get('opacity_cull_threshold', 0.005)

        # Geometry constraint parameters
        scale_loss_weight = config.get('scale_loss_weight', 150.0)
        single_view_weight = config.get('single_view_weight', 0.1)
        single_view_weight_from_iter = config.get('single_view_weight_from_iter', 1000)
        multi_view_geo_weight = config.get('multi_view_geo_weight', 0.2)
        multi_view_ncc_weight = config.get('multi_view_ncc_weight', 0.3)
        multi_view_weight_from_iter = config.get('multi_view_weight_from_iter', 1500)
        multi_view_pixel_noise_th = config.get('multi_view_pixel_noise_th', 1.0)

        # Print actual parameters being used
        print(f"PGSR Training Parameters:")
        print(f"  iterations: {iterations}")
        print(f"  densify_abs_grad_threshold: {densify_abs_grad_threshold}")
        print(f"  max_abs_split_points: {max_abs_split_points}")
        print(f"  opacity_cull_threshold: {opacity_cull_threshold}")
        print(f"  scale_loss_weight: {scale_loss_weight}")
        print(f"  single_view_weight: {single_view_weight}")
        print(f"  single_view_weight_from_iter: {single_view_weight_from_iter}")
        print(f"  multi_view_geo_weight: {multi_view_geo_weight}")
        print(f"  multi_view_ncc_weight: {multi_view_ncc_weight}")
        print(f"  multi_view_weight_from_iter: {multi_view_weight_from_iter}")
        print(f"  multi_view_pixel_noise_th: {multi_view_pixel_noise_th}")

        # Use absolute paths since train.py runs in external/PGSR directory
        abs_data_path = Path(data_path).absolute()
        abs_output_path = Path(output_path).absolute()

        cmd = f"""python -u train.py \
            -s {abs_data_path} \
            -m {abs_output_path} \
            -r 1 \
            --iterations {iterations} \
            --densify_abs_grad_threshold {densify_abs_grad_threshold} \
            --max_abs_split_points {max_abs_split_points} \
            --opacity_cull_threshold {opacity_cull_threshold} \
            --scale_loss_weight {scale_loss_weight} \
            --single_view_weight {single_view_weight} \
            --single_view_weight_from_iter {single_view_weight_from_iter} \
            --multi_view_geo_weight {multi_view_geo_weight} \
            --multi_view_ncc_weight {multi_view_ncc_weight} \
            --multi_view_weight_from_iter {multi_view_weight_from_iter} \
            --multi_view_pixel_noise_th {multi_view_pixel_noise_th} \
            --eval \
            --white_background"""

        result = self.run_command(cmd, log_output=True, log_dir=str(abs_output_path))

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        return True

    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """Extract mesh from PGSR model"""
        from pathlib import Path

        config = self.get_default_config()
        config.update(kwargs)

        iteration = config.get('iterations', 30000)

        # Use absolute paths since render.py runs in external/PGSR directory
        abs_model_path = Path(model_path).absolute()

        # Get data path
        data_path = kwargs.get('data_path')
        if not data_path:
            # Try to infer
            parts = abs_model_path.parts
            if 'models' in parts:
                idx = parts.index('models')
                object_name = parts[idx + 1] if idx + 1 < len(parts) else None
                scene_name = parts[idx + 2] if idx + 2 < len(parts) else None
                if object_name and scene_name:
                    data_path = f"../datasets/openmaterial/{object_name}/{scene_name}"

        # Convert data_path to absolute if provided
        if data_path:
            abs_data_path = Path(data_path).absolute()
        else:
            abs_data_path = data_path

        # PGSR render.py extracts mesh automatically when --skip_train is NOT set
        # Mesh is saved to {model_path}/mesh/tsdf_fusion_post.ply
        voxel_size = config.get('voxel_size', 0.004)
        max_depth = config.get('max_depth', 5.0)
        num_cluster = config.get('num_cluster', 1)
        use_depth_filter = config.get('use_depth_filter', True)

        depth_filter_flag = "--use_depth_filter" if use_depth_filter else ""
        cmd = f"""python render.py \
            -s {abs_data_path} \
            -m {abs_model_path} \
            --iteration {iteration} \
            --voxel_size {voxel_size} \
            --max_depth {max_depth} \
            --num_cluster {num_cluster} \
            {depth_filter_flag} \
            --skip_test"""

        result = self.run_command(cmd)

        if result.returncode != 0:
            print(f"Mesh extraction failed: {result.stderr}")
            return False

        # Copy mesh to output path
        # PGSR saves mesh at: {model_path}/mesh/tsdf_fusion_post.ply
        mesh_dir = abs_model_path / "mesh"

        # Try post-processed mesh first
        source_mesh = mesh_dir / "tsdf_fusion_post.ply"
        if not source_mesh.exists():
            # Fall back to regular mesh
            source_mesh = mesh_dir / "tsdf_fusion.ply"

        if source_mesh.exists():
            import shutil
            shutil.copy(source_mesh, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {mesh_dir}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default PGSR configuration

        Note: use_depth_filter is disabled due to incorrect ray direction in world space
        Enhanced geometry constraints for better depth quality
        """
        return {
            'iterations': 30000,
            'densify_abs_grad_threshold': 0.0008,
            'max_abs_split_points': 50000,
            'opacity_cull_threshold': 0.005,
            # Geometry constraint parameters (enhanced for OpenMaterial)
            'scale_loss_weight': 150.0,  # Force Gaussians to be flat (original: 100.0)
            'single_view_weight': 0.1,  # Single-view normal constraint (original: 0.015)
            'single_view_weight_from_iter': 1000,  # Enable early (original: 7000)
            'multi_view_geo_weight': 0.2,  # Multi-view geometry constraint (original: 0.03)
            'multi_view_ncc_weight': 0.3,  # Multi-view photometric constraint (original: 0.15)
            'multi_view_weight_from_iter': 1500,  # Enable early (original: 7000)
            'multi_view_pixel_noise_th': 1.0,  # Pixel noise threshold
            # Mesh extraction parameters
            'voxel_size': 0.004,
            'max_depth': 5.0,
            'num_cluster': 1,
            'use_depth_filter': False,  # Disabled: get_rays() returns camera-space rays, not world-space
        }
