"""
2DGS (2D Gaussian Splatting) Method Wrapper
"""

from pathlib import Path
from typing import Dict, Any
from ..base_method import BaseMethod


class TwoDGSMethod(BaseMethod):
    """Wrapper for 2DGS method"""

    def __init__(self, repo_path: str = "external/2DGS"):
        super().__init__(
            method_name="2dgs",
            repo_path=repo_path,
            conda_env="surfel_splatting"
        )

    def setup(self) -> bool:
        """Setup 2DGS environment"""
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
                "pip install torch==2.3.1 torchvision==0.18.1 "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
            if result.returncode != 0:
                return False

        # Install dependencies
        result = self.run_command("pip install plyfile tqdm opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if result.returncode != 0:
            return False

        # Build CUDA extensions
        print("Building CUDA extensions...")
        result = self.run_command("pip install submodules/diff-surfel-rasterization")
        if result.returncode != 0:
            print(f"Failed to build diff-surfel-rasterization: {result.stderr}")
            return False

        result = self.run_command("pip install submodules/simple-knn")
        if result.returncode != 0:
            print(f"Failed to build simple-knn: {result.stderr}")
            return False

        print("✓ 2DGS setup complete")
        return True

    def convert_data(self, input_path: str, output_path: str) -> bool:
        """2DGS uses transforms.json format directly, create symlink to avoid data duplication"""
        import os
        from pathlib import Path

        output_path_obj = Path(output_path)
        input_path_obj = Path(input_path)

        print(f"Creating symlink: {output_path_obj} -> {input_path_obj}")

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

        # Verify the symlink works
        if not output_path_obj.exists():
            print(f"Error: Symlink created but target doesn't exist!")
            return False

        transforms_file = output_path_obj / "transforms_train.json"
        print(f"Checking for transforms file: {transforms_file}")
        print(f"Transforms file exists: {transforms_file.exists()}")

        return True

    def train(self, data_path: str, output_path: str, **kwargs) -> bool:
        """Train 2DGS"""
        from pathlib import Path

        config = self.get_default_config()
        config.update(kwargs)

        iterations = config.get('iterations', 30000)
        lambda_normal = config.get('lambda_normal', 0.05)
        lambda_dist = config.get('lambda_dist', 1000)
        depth_ratio = config.get('depth_ratio', 0)

        # Use absolute paths since train.py runs in external/2DGS directory
        abs_data_path = Path(data_path).absolute()
        abs_output_path = Path(output_path).absolute()

        cmd = f"""python train.py \
            -s {abs_data_path} \
            -m {abs_output_path} \
            -r 1 \
            --iterations {iterations} \
            --lambda_normal {lambda_normal} \
            --lambda_dist {lambda_dist} \
            --depth_ratio {depth_ratio}"""

        result = self.run_command(cmd, log_output=True, log_dir=str(abs_output_path))

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        return True

    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """Extract mesh from 2DGS model"""
        config = self.get_default_config()
        config.update(kwargs)

        mesh_res = config.get('mesh_res', 1024)
        depth_ratio = config.get('depth_ratio', 0)

        # Find the data path from model path
        # In 2DGS, we need the original data path for mesh extraction
        # This is a limitation - we'll need to pass it through kwargs
        data_path = kwargs.get('data_path')
        if not data_path:
            print("Warning: data_path not provided in kwargs, mesh extraction may fail")
            # Try to infer from model path structure
            # Assume structure: output_dir/models/object/scene
            # Data at: datasets/openmaterial/object/scene
            parts = Path(model_path).parts
            if 'models' in parts:
                idx = parts.index('models')
                object_name = parts[idx + 1] if idx + 1 < len(parts) else None
                scene_name = parts[idx + 2] if idx + 2 < len(parts) else None
                if object_name and scene_name:
                    data_path = f"../datasets/openmaterial/{object_name}/{scene_name}"

        cmd = f"""python render.py \
            -s {data_path} \
            -m {model_path} \
            --skip_test \
            --skip_train \
            --mesh_res {mesh_res} \
            --depth_ratio {depth_ratio}"""

        result = self.run_command(cmd)

        if result.returncode != 0:
            print(f"Mesh extraction failed: {result.stderr}")
            return False

        # Copy mesh to output path
        source_mesh = Path(model_path) / "fuse.ply"
        if source_mesh.exists():
            import shutil
            shutil.copy(source_mesh, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {source_mesh}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default 2DGS configuration"""
        return {
            'iterations': 30000,
            'lambda_normal': 0.05,
            'lambda_dist': 1000,
            'depth_ratio': 0,
            'mesh_res': 1024,
        }
