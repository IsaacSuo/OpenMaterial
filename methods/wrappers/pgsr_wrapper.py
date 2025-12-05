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
        print("Installing dependencies...")
        result = self.run_command(
            "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        if result.returncode != 0:
            print(f"Failed to install base dependencies: {result.stderr}")
            return False

        # Install PyTorch3D (required for PGSR mesh processing)
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

        # Build CUDA extensions
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

        # Use absolute paths since train.py runs in external/PGSR directory
        abs_data_path = Path(data_path).absolute()
        abs_output_path = Path(output_path).absolute()

        cmd = f"""python train.py \
            -s {abs_data_path} \
            -m {abs_output_path} \
            -r 1 \
            --iterations {iterations}"""

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

        cmd = f"""python render.py \
            -s {abs_data_path} \
            -m {abs_model_path} \
            --iteration {iteration} \
            --skip_test \
            --skip_train \
            --compute_pcd \
            --compute_mesh"""

        result = self.run_command(cmd)

        if result.returncode != 0:
            print(f"Mesh extraction failed: {result.stderr}")
            return False

        # Copy mesh to output path
        mesh_dir = abs_model_path / "point_cloud" / f"iteration_{iteration}"

        # Try post-processed mesh first
        source_mesh = mesh_dir / "fuse_post.ply"
        if not source_mesh.exists():
            # Fall back to regular mesh
            source_mesh = mesh_dir / "fuse.ply"

        if source_mesh.exists():
            import shutil
            shutil.copy(source_mesh, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {mesh_dir}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default PGSR configuration"""
        return {
            'iterations': 30000,
        }
