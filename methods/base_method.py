"""
Base class for all reconstruction methods
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any
import subprocess
import os


class BaseMethod(ABC):
    """Abstract base class for 3D reconstruction methods"""

    def __init__(self, method_name: str, repo_path: str, conda_env: str):
        """
        Initialize method

        Args:
            method_name: Name of the method
            repo_path: Path to the method's repository
            conda_env: Name of the conda environment
        """
        self.method_name = method_name
        self.repo_path = Path(repo_path)
        self.conda_env = conda_env

        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository not found at {self.repo_path}")

    @abstractmethod
    def setup(self) -> bool:
        """
        Setup method environment (install dependencies, compile CUDA code, etc.)

        Returns:
            bool: True if setup successful
        """
        pass

    @abstractmethod
    def convert_data(self, input_path: str, output_path: str) -> bool:
        """
        Convert OpenMaterial data format to method-specific format

        Args:
            input_path: Path to OpenMaterial scene
            output_path: Path to save converted data

        Returns:
            bool: True if conversion successful
        """
        pass

    @abstractmethod
    def train(self, data_path: str, output_path: str, **kwargs) -> bool:
        """
        Train the method on given data

        Args:
            data_path: Path to training data
            output_path: Path to save outputs
            **kwargs: Method-specific parameters

        Returns:
            bool: True if training successful
        """
        pass

    @abstractmethod
    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """
        Extract mesh from trained model

        Args:
            model_path: Path to trained model
            output_mesh_path: Path to save mesh
            **kwargs: Method-specific parameters

        Returns:
            bool: True if extraction successful
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the method

        Returns:
            Dict: Default configuration
        """
        pass

    def run_command(self, cmd: str, cwd: Optional[str] = None,
                   use_conda: bool = True, log_output: bool = False,
                   log_dir: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Run a shell command

        Args:
            cmd: Command to run
            cwd: Working directory
            use_conda: Whether to activate conda environment
            log_output: If True, stream output to terminal and save to log files
            log_dir: Directory to save logs (required if log_output=True)

        Returns:
            CompletedProcess: Result of the command
        """
        if cwd is None:
            cwd = str(self.repo_path)

        if use_conda:
            # Activate conda environment and run command
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env} && {cmd}"
        else:
            full_cmd = cmd

        if log_output and log_dir:
            # Stream output to terminal and save to log files
            import sys
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            stdout_log = log_dir_path / "training.log"
            stderr_log = log_dir_path / "training_error.log"

            print(f"Training logs: {stdout_log}")

            # Use Popen for real-time output
            process = subprocess.Popen(
                full_cmd,
                shell=True,
                cwd=cwd,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            stdout_lines = []
            stderr_lines = []

            # Open log files
            with open(stdout_log, 'w') as f_out, open(stderr_log, 'w') as f_err:
                # Read output in real-time
                while True:
                    # Read stdout
                    if process.stdout:
                        stdout_line = process.stdout.readline()
                        if stdout_line:
                            print(stdout_line, end='', flush=True)
                            f_out.write(stdout_line)
                            f_out.flush()
                            stdout_lines.append(stdout_line)

                    # Read stderr
                    if process.stderr:
                        stderr_line = process.stderr.readline()
                        if stderr_line:
                            print(stderr_line, end='', file=sys.stderr, flush=True)
                            f_err.write(stderr_line)
                            f_err.flush()
                            stderr_lines.append(stderr_line)

                    # Check if process finished
                    if process.poll() is not None:
                        # Read remaining output
                        if process.stdout:
                            for line in process.stdout:
                                print(line, end='', flush=True)
                                f_out.write(line)
                                stdout_lines.append(line)
                        if process.stderr:
                            for line in process.stderr:
                                print(line, end='', file=sys.stderr, flush=True)
                                f_err.write(line)
                                stderr_lines.append(line)
                        break

            # Create CompletedProcess-like object for compatibility
            class LoggedResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = ''.join(stdout)
                    self.stderr = ''.join(stderr)

            return LoggedResult(process.returncode, stdout_lines, stderr_lines)
        else:
            # Original behavior: capture output
            result = subprocess.run(
                full_cmd,
                shell=True,
                cwd=cwd,
                executable='/bin/bash',
                capture_output=True,
                text=True
            )
            return result

    def check_environment(self) -> bool:
        """
        Check if conda environment exists

        Returns:
            bool: True if environment exists
        """
        result = subprocess.run(
            f"conda env list | grep {self.conda_env}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0

    def process_scene(self, input_scene: str, output_dir: str,
                     gpu_id: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Process a single scene (convert, train, extract mesh)

        Args:
            input_scene: Path to OpenMaterial scene
            output_dir: Base output directory
            gpu_id: GPU ID to use
            **kwargs: Method-specific parameters

        Returns:
            Dict: Results including paths and metrics
        """
        scene_name = Path(input_scene).name
        object_name = Path(input_scene).parent.name

        # Setup paths
        converted_data = Path(output_dir) / "converted_data" / object_name / scene_name
        model_output = Path(output_dir) / "models" / object_name / scene_name
        mesh_output = Path(output_dir) / "meshes" / object_name / f"{scene_name}.ply"

        converted_data.parent.mkdir(parents=True, exist_ok=True)
        model_output.parent.mkdir(parents=True, exist_ok=True)
        mesh_output.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'scene': scene_name,
            'object': object_name,
            'success': False,
            'converted_data': str(converted_data),
            'model_output': str(model_output),
            'mesh_output': str(mesh_output),
        }

        try:
            # Step 1: Convert data
            print(f"[{self.method_name}] Converting data for {scene_name}...")
            if not self.convert_data(input_scene, str(converted_data)):
                results['error'] = 'Data conversion failed'
                return results

            # Step 2: Train
            print(f"[{self.method_name}] Training on {scene_name}...")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if not self.train(str(converted_data), str(model_output), **kwargs):
                results['error'] = 'Training failed'
                return results

            # Step 3: Extract mesh
            print(f"[{self.method_name}] Extracting mesh for {scene_name}...")
            if not self.extract_mesh(str(model_output), str(mesh_output), **kwargs):
                results['error'] = 'Mesh extraction failed'
                return results

            results['success'] = True
            print(f"[{self.method_name}] ✓ Successfully processed {scene_name}")

        except Exception as e:
            results['error'] = str(e)
            print(f"[{self.method_name}] ✗ Error processing {scene_name}: {e}")

        return results
