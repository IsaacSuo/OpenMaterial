"""
NeuS2 Method Wrapper
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from ..base_method import BaseMethod


class NeuS2Method(BaseMethod):
    """Wrapper for NeuS2 method"""

    def __init__(self, repo_path: str = "external/NeuS2"):
        super().__init__(
            method_name="neus2",
            repo_path=repo_path,
            conda_env="neus2"
        )

    def setup(self) -> bool:
        """Setup NeuS2 environment"""
        if not self.check_environment():
            print(f"Creating conda environment: {self.conda_env}")
            result = self.run_command(
                f"conda create -n {self.conda_env} python=3.9 -y",
                use_conda=False
            )
            if result.returncode != 0:
                print(f"Failed to create environment: {result.stderr}")
                return False

        # Install dependencies
        print("Installing dependencies...")
        result = self.run_command(
            "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        if result.returncode != 0:
            print(f"Failed to install dependencies: {result.stderr}")
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
                print(f"Failed to install PyTorch: {result.stderr}")
                return False

        # Install PyTorch3D (required for NeuS2)
        print("Installing PyTorch3D...")
        # Try prebuilt version from PyPI mirror first (faster)
        result = self.run_command(
            "pip install pytorch3d -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        if result.returncode != 0:
            print(f"⚠ PyPI version failed, trying from source...")
            # Try from source if prebuilt fails
            result = self.run_command(
                'pip install "git+https://github.com/facebookresearch/pytorch3d.git"'
            )
            if result.returncode != 0:
                print(f"Failed to install PyTorch3D: {result.stderr}")
                return False

        # Build CUDA code
        print("Building CUDA code...")
        result = self.run_command("cmake . -B build", use_conda=False)
        if result.returncode != 0:
            print(f"CMake configuration failed: {result.stderr}")
            return False

        result = self.run_command("cmake --build build --config RelWithDebInfo -j", use_conda=False)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False

        # Check if testbed was built
        testbed_path = self.repo_path / "build" / "testbed"
        if not testbed_path.exists():
            print("Build succeeded but testbed not found")
            return False

        print("✓ NeuS2 setup complete")
        return True

    def convert_data(self, input_path: str, output_path: str) -> bool:
        """Convert OpenMaterial data to NeuS2 format"""
        converter_script = self.repo_path / "tools" / "convert_openmaterial.py"

        if not converter_script.exists():
            print(f"Converter script not found at {converter_script}")
            return False

        # Use absolute paths to avoid issues with working directory
        abs_input = Path(input_path).absolute()
        abs_output = Path(output_path).absolute()

        cmd = f"python {converter_script.absolute()} --input {abs_input} --output {abs_output} --splits train test"
        print(f"Running conversion: {cmd}")
        result = self.run_command(cmd)

        print(f"Conversion stdout: {result.stdout}")
        print(f"Conversion stderr: {result.stderr}")
        print(f"Conversion returncode: {result.returncode}")

        if result.returncode != 0:
            print(f"Data conversion failed with return code {result.returncode}")
            return False

        # Verify output files were created
        output_train = abs_output / "transforms_train.json"
        if not output_train.exists():
            print(f"Error: Expected output file not found: {output_train}")
            print(f"Conversion command output: {result.stdout}")
            return False

        return True

    def train(self, data_path: str, output_path: str, **kwargs) -> bool:
        """Train NeuS2"""
        config = self.get_default_config()
        config.update(kwargs)

        n_steps = config.get('n_steps', 15000)
        network = config.get('network', 'dtu.json')
        scene_file = Path(data_path) / "transforms_train.json"

        if not scene_file.exists():
            print(f"Scene file not found: {scene_file}")
            return False

        exp_name = Path(output_path).name

        cmd = f"""python scripts/run.py \
            --scene {scene_file} \
            --name {exp_name} \
            --network {network} \
            --n_steps {n_steps}"""

        result = self.run_command(cmd)

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        return True

    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """Extract mesh from NeuS2 model"""
        config = self.get_default_config()
        config.update(kwargs)

        n_steps = config.get('n_steps', 15000)
        marching_cubes_res = config.get('marching_cubes_res', 512)

        # Find the trained model
        exp_name = Path(model_path).name
        output_dir = self.repo_path / "output" / exp_name
        checkpoint_dir = output_dir / "checkpoints"

        if not checkpoint_dir.exists():
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            return False

        # NeuS2 saves mesh during training, just copy it
        source_mesh = output_dir / "mesh" / f"mesh_{n_steps}.ply"
        if source_mesh.exists():
            import shutil
            shutil.copy(source_mesh, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {source_mesh}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default NeuS2 configuration"""
        return {
            'n_steps': 15000,
            'network': 'dtu.json',
            'marching_cubes_res': 512,
        }
