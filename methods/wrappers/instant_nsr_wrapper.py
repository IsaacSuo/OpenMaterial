"""
Instant-NSR-PL Method Wrapper (Baseline)
"""

from pathlib import Path
from typing import Dict, Any
from ..base_method import BaseMethod


class InstantNSRMethod(BaseMethod):
    """Wrapper for Instant-NSR-PL method (baseline)"""

    def __init__(self, repo_path: str = "instant-nsr-pl"):
        super().__init__(
            method_name="instant-nsr-pl",
            repo_path=repo_path,
            conda_env="instant-nsr-pl"  # Assume existing environment
        )

    def setup(self) -> bool:
        """Setup Instant-NSR-PL environment"""
        # Instant-NSR-PL is already set up in the project
        print("✓ Instant-NSR-PL already configured")
        return True

    def convert_data(self, input_path: str, output_path: str) -> bool:
        """No conversion needed for Instant-NSR-PL"""
        return True

    def train(self, data_path: str, output_path: str, **kwargs) -> bool:
        """Train Instant-NSR-PL"""
        config = self.get_default_config()
        config.update(kwargs)

        # Use absolute paths since launch.py runs in instant-nsr-pl directory
        abs_data_path = Path(data_path).absolute()
        abs_output_path = Path(output_path).absolute()

        # Extract scene information
        scene_name = abs_data_path.name
        object_name = abs_data_path.parent.name

        # Get material name from scene name
        material_name = scene_name.split('-')[-1] if '-' in scene_name else 'diffuse'

        # Run training using launch.py
        cmd = f"""python launch.py \
            --config configs/neus-openmaterial-wmask.yaml \
            --output_dir {abs_output_path} \
            --train \
            dataset.bsdf_name={material_name} \
            dataset.object={object_name} \
            dataset.scene={scene_name} \
            dataset.root_dir={abs_data_path} \
            trial_name={object_name}"""

        result = self.run_command(cmd, log_output=True, log_dir=str(abs_output_path))

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        return True

    def extract_mesh(self, model_path: str, output_mesh_path: str, **kwargs) -> bool:
        """Extract mesh from Instant-NSR-PL model"""
        # Use absolute paths
        abs_model_path = Path(model_path).absolute()

        # Instant-NSR-PL generates mesh during training
        # Find the mesh in the output directory
        mesh_file = abs_model_path / "mesh" / "mesh.ply"

        if not mesh_file.exists():
            # Try alternative location
            exp_dir = abs_model_path.parent
            mesh_file = exp_dir / "mesh" / "mesh.ply"

        if mesh_file.exists():
            import shutil
            shutil.copy(mesh_file, output_mesh_path)
            return True
        else:
            print(f"Mesh not found at {abs_model_path}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """Get default Instant-NSR-PL configuration"""
        return {
            'config': 'configs/neus-openmaterial-wmask.yaml',
        }
