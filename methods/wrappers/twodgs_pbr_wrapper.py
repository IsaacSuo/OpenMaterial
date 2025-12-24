"""
2DGS-PBR (2D Gaussian Splatting with PBR) Method Wrapper
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
from ..base_method import BaseMethod


class TwoDGSPBRMethod(BaseMethod):
    """Wrapper for 2DGS-PBR method (PBR-enhanced 2D Gaussian Splatting)"""

    def __init__(self, repo_path: str = "external/2DGS-PBR"):
        super().__init__(
            method_name="2dgs-pbr",
            repo_path=repo_path,
            conda_env=""  # Not using conda
        )

    def setup(self) -> bool:
        """Setup 2DGS-PBR environment

        Verifies that required dependencies are available in current Python environment.
        """
        # Verify CUDA extensions are available
        result = subprocess.run(
            "python -c \"import diff_surfel_rasterization; import simple_knn\"",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("CUDA extensions not found. Please build them:")
            print("  cd external/2DGS-PBR")
            print("  pip install submodules/diff-surfel-rasterization")
            print("  pip install submodules/simple-knn")
            return False

        print("2DGS-PBR environment ready")
        return True

    def run_command(self, cmd: str, cwd=None, use_conda: bool = False,
                   log_output: bool = False, log_dir=None):
        """Run command without conda activation"""
        if cwd is None:
            cwd = str(self.repo_path)

        if log_output and log_dir:
            import re
            import time
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            stdout_log = log_dir_path / "training.log"
            print(f"Training logs: {stdout_log}")

            with open(stdout_log, 'w') as f_out:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    executable='/bin/bash',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                progress_patterns = [
                    r'Iteration.*?(\d+)',
                    r'Iter.*?(\d+)',
                    r'Step.*?(\d+)',
                    r'Epoch.*?(\d+)',
                    r'\d+%',
                    r'Loss',
                    r'PSNR',
                ]

                last_progress = None
                last_update_time = time.time()

                for line in process.stdout:
                    f_out.write(line)
                    f_out.flush()

                    current_time = time.time()
                    is_progress = any(re.search(p, line, re.IGNORECASE) for p in progress_patterns)

                    if is_progress and (current_time - last_update_time > 0.5):
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
                        print(f"\r{clean_line[:120]:<120}", end='', flush=True)
                        last_progress = clean_line
                        last_update_time = current_time
                    elif any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                        print(f"\n{line.strip()}")

                returncode = process.wait()

                if last_progress:
                    print(f"\r{last_progress[:120]:<120}")
                print(f"\nTraining completed with return code: {returncode}")

            with open(stdout_log, 'r') as f:
                stdout_content = f.read()

            class Result:
                def __init__(self, rc, out, err):
                    self.returncode = rc
                    self.stdout = out
                    self.stderr = err

            return Result(returncode, stdout_content, "")
        else:
            return subprocess.run(
                cmd, shell=True, cwd=cwd, executable='/bin/bash',
                capture_output=True, text=True
            )

    def convert_data(self, input_path: str, output_path: str) -> bool:
        """2DGS-PBR uses transforms.json format directly, create symlink to avoid data duplication"""
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
        """Train 2DGS-PBR"""
        from pathlib import Path

        config = self.get_default_config()
        config.update(kwargs)

        iterations = config.get('iterations', 30000)
        lambda_normal = config.get('lambda_normal', 0.0)
        lambda_dist = config.get('lambda_dist', 0.0)
        depth_ratio = config.get('depth_ratio', 0)

        # Memory optimization parameters
        densify_until_iter = config.get('densify_until_iter', 12000)
        densify_grad_threshold = config.get('densify_grad_threshold', 0.0003)

        # PBR specific parameters
        use_pbr = config.get('use_pbr', False)

        # Print actual parameters being used
        print(f"2DGS-PBR Training Parameters:")
        print(f"  iterations: {iterations}")
        print(f"  densify_until_iter: {densify_until_iter}")
        print(f"  densify_grad_threshold: {densify_grad_threshold}")
        print(f"  use_pbr: {use_pbr}")

        # Use absolute paths since train.py runs in external/2DGS-PBR directory
        abs_data_path = Path(data_path).absolute()
        abs_output_path = Path(output_path).absolute()

        cmd = f"""python -u train.py \
            -s {abs_data_path} \
            -m {abs_output_path} \
            -r 1 \
            --iterations {iterations} \
            --lambda_normal {lambda_normal} \
            --lambda_dist {lambda_dist} \
            --depth_ratio {depth_ratio} \
            --densify_until_iter {densify_until_iter} \
            --densify_grad_threshold {densify_grad_threshold} \
            --eval \
            --white_background"""

        # Add PBR flag if enabled (will be implemented in later phases)
        if use_pbr:
            cmd += " --use_pbr"

        result = self.run_command(cmd, log_output=True, log_dir=str(abs_output_path))

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        return True

    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """Extract mesh from 2DGS-PBR model"""
        from pathlib import Path

        config = self.get_default_config()
        config.update(kwargs)

        mesh_res = config.get('mesh_res', 1024)
        depth_ratio = config.get('depth_ratio', 0)

        # Use absolute paths since render.py runs in external/2DGS-PBR directory
        abs_model_path = Path(model_path).absolute()

        # Find the data path from model path
        data_path = kwargs.get('data_path')
        if not data_path:
            print("Warning: data_path not provided in kwargs, mesh extraction may fail")
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

        iteration = config.get('iterations', 30000)

        cmd = f"""python render.py \
            -s {abs_data_path} \
            -m {abs_model_path} \
            --iteration {iteration} \
            --skip_test \
            --skip_train \
            --mesh_res {mesh_res} \
            --depth_ratio {depth_ratio}"""

        result = self.run_command(cmd)

        if result.returncode != 0:
            print(f"Mesh extraction failed: {result.stderr}")
            return False

        # Copy mesh to output path
        train_dir = abs_model_path / "train" / f"ours_{iteration}"

        # Try post-processed mesh first
        source_mesh = train_dir / "fuse_post.ply"
        if not source_mesh.exists():
            source_mesh = train_dir / "fuse.ply"

        if source_mesh.exists():
            import shutil
            shutil.copy(source_mesh, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {train_dir}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default 2DGS-PBR configuration

        Uses official 2DGS default parameters for best reconstruction quality.
        PBR-specific parameters added for material estimation.
        """
        return {
            'iterations': 30000,
            'lambda_normal': 0.05,  # Normal consistency regularization
            'lambda_dist': 0.0,
            'depth_ratio': 0,
            'mesh_res': 1024,
            # Official 2DGS densification parameters
            'densify_until_iter': 15000,
            'densify_grad_threshold': 0.0002,
            # PBR specific parameters
            'use_pbr': False,  # Enable PBR mode
        }
