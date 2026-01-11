# 2DGS-PBR Architecture and Integration Guide

## Executive Summary

2DGS-PBR is an extension of 2D Gaussian Splatting that adds **Physically Based Rendering (PBR)** capabilities. It represents scenes with 2D oriented disks (surface elements/surfels) and learns PBR material properties (albedo, roughness, metallic) alongside geometric and appearance parameters. The system uses deferred rendering with G-Buffers to enable physically-based shading with environment lighting.

**Key Innovation**: Integrates PBR material properties into the differentiable rasterization pipeline, enabling end-to-end training of both geometry and material properties.

---

## 1. Project Structure Overview

```
2DGS-PBR/
├── arguments/
│   └── __init__.py              # Configuration parameter classes
├── gaussian_renderer/
│   ├── __init__.py              # Core rendering pipeline with G-Buffer support
│   └── network_gui.py           # Interactive viewer interface
├── scene/
│   ├── __init__.py              # Scene class
│   ├── gaussian_model.py        # GaussianModel with PBR support
│   ├── cameras.py               # Camera class for view management
│   ├── dataset_readers.py       # Data loading (COLMAP, Blender)
│   └── colmap_loader.py         # COLMAP format parser
├── utils/
│   ├── pbr_utils.py             # PBR shading (Cook-Torrance BRDF, environment lighting)
│   ├── loss_utils.py            # Loss functions including PBR regularization
│   ├── image_utils.py           # Image metrics (PSNR, gradient maps)
│   ├── graphics_utils.py        # Camera transformations, projection matrices
│   ├── general_utils.py         # Utilities (logging, lr scheduling)
│   ├── sh_utils.py              # Spherical harmonics (SH) basis evaluation
│   ├── camera_utils.py          # Camera intrinsics/extrinsics handling
│   └── [other utils]
├── submodules/
│   ├── diff-surfel-rasterization/  # CUDA rasterizer (2D surfel rendering)
│   └── simple-knn/                 # KNN for Gaussian initialization
├── train_pbr.py                 # Main PBR training script
├── render_pbr.py                # Rendering and G-Buffer visualization
├── train.py                     # Standard training (non-PBR)
├── render.py                    # Standard rendering
└── environment.yml              # Conda environment specification
```

---

## 2. Core Modules and Responsibilities

### 2.1 Arguments Module (`arguments/__init__.py`)

Defines configuration parameter groups:

**ModelParams**
- `sh_degree`: Spherical harmonics degree (default: 3)
- `source_path`: Path to COLMAP/Blender dataset
- `model_path`: Output path for trained model
- `images`: Image subdirectory name (default: "images")
- `resolution`: Image resolution (-1 = use original)
- `white_background`: Background color flag
- `render_items`: Items to render in viewer

**PipelineParams**
- `convert_SHs_python`: SH to RGB conversion on CPU (vs GPU)
- `compute_cov3D_python`: Covariance computation on CPU
- `depth_ratio`: 0 for expected depth, 1 for median (default: 0)
- `debug`: Debug mode flag

**OptimizationParams**
- Learning rates: position, feature, opacity, scaling, rotation, feature_rest
- Densification settings: intervals, thresholds, iteration ranges
- Loss weights: `lambda_dssim`, `lambda_dist`, `lambda_normal`, `lambda_pbr`, `lambda_pbr_reg`
- Material-specific LRs: albedo_lr, roughness_lr, metallic_lr

### 2.2 Gaussian Renderer (`gaussian_renderer/__init__.py`)

**Primary Function**: `render(viewpoint_camera, pc, pipe, bg_color, render_pbr=False)`

Performs differentiable rasterization with optional G-Buffer generation.

**Key Features**:
- Converts Gaussian parameters to 2D projections
- Renders SH-based appearance via rasterizer
- Extracts normal and depth maps
- **G-Buffer Rendering** (when `render_pbr=True`):
  - Albedo map: [3, H, W]
  - Roughness map: [1, H, W]
  - Metallic map: [1, H, W]

**Return Dictionary Keys**:
- `render`: [3, H, W] final RGB image
- `viewspace_points`: Screen-space point positions
- `visibility_filter`: Mask of visible Gaussians
- `radii`: 2D radii in screen space
- `rend_alpha`: Opacity map
- `rend_normal`: World-space surface normal
- `surf_depth`: Surface depth (expected or median)
- `rend_dist`: Depth distortion regularization
- `gbuffer_albedo`, `gbuffer_roughness`, `gbuffer_metallic`: (if PBR enabled)

### 2.3 Scene Management (`scene/__init__.py`)

**Scene Class**:
- Manages camera lists, Gaussian model initialization
- Loads data (COLMAP or Blender format)
- Provides `getTrainCameras()` and `getTestCameras()`
- Handles model saving/loading

**Data Format Requirements**:
- **COLMAP**: `sparse/` directory with camera intrinsics/extrinsics
- **Blender/NeRF**: `transforms_train.json`, `transforms_test.json`
- Point cloud: PLY file with RGB colors

### 2.4 Gaussian Model (`scene/gaussian_model.py`)

**Core Class**: `GaussianModel(sh_degree, use_pbr=False)`

**Parameters** (per Gaussian point):
- Geometry: position (xyz), covariance (scaling + rotation)
- Appearance: SH coefficients (DC + higher orders)
- Opacity: single value [0, 1]
- **PBR (when enabled)**:
  - `_albedo`: [N, 3] base color
  - `_roughness`: [N, 1] roughness in [0.1, 0.999]
  - `_metallic`: [N, 1] metallicness in [0, 1]

**Key Methods**:
- `create_from_pcd()`: Initialize from point cloud
- `training_setup()`: Create optimizer with all learnable parameters
- `densify_and_prune()`: Adaptive point cloud densification
- `save_ply()` / `load_ply()`: Persistence with PBR parameters
- Property getters with activations: `get_albedo`, `get_roughness`, `get_metallic`

**Activation Functions**:
- Albedo: `sigmoid(raw) -> [0, 1]`
- Roughness: `clamp(sigmoid(raw), 0.1, 0.999)`
- Metallic: `sigmoid(raw) -> [0, 1]`

### 2.5 PBR Utilities (`utils/pbr_utils.py`)

**EnvironmentLight Class**:
- Loads HDR environment maps (.hdr, .exr, .png)
- Learnable parameters for environment lighting
- Methods:
  - `sample(directions)`: Sample environment color given world directions
  - `forward(directions)`: Alias for sampling

**PBR Shading Functions**:

1. **Cook-Torrance BRDF Functions**:
   - `fresnel_schlick()`: Fresnel approximation
   - `distribution_ggx()`: GGX normal distribution
   - `geometry_schlick_ggx()`: Schlick-GGX geometry function
   - `geometry_smith()`: Combined geometry shadowing

2. **Shading Methods**:
   - `pbr_shading()`: Single light PBR with Cook-Torrance BRDF
   - `pbr_shading_env()`: Environment lighting with importance sampling
   - `screen_space_pbr_shading()`: Deferred shading from G-Buffer

**Screen-Space Shading Pipeline**:
```
G-Buffer inputs:
  - gbuffer_albedo [3, H, W]
  - gbuffer_roughness [1, H, W]
  - gbuffer_metallic [1, H, W]
  - gbuffer_normal [3, H, W]
  - gbuffer_depth [1, H, W]
  - camera_center, camera_transform
  - env_light (EnvironmentLight)

Output: shaded_image [3, H, W]

Process:
  1. Extract view direction for each pixel
  2. Normalize normals
  3. Sample environment map at reflection direction (specular)
  4. Sample environment map at normal direction (diffuse)
  5. Apply Fresnel term and energy conservation
  6. Combine diffuse + specular
```

### 2.6 Loss Functions (`utils/loss_utils.py`)

**Reconstruction Losses**:
- `l1_loss()`: L1 pixel error
- `ssim()`: Structural similarity

**PBR Losses**:
- `pbr_reconstruction_loss()`: L1 + SSIM between shaded and GT
- `material_smoothness_loss()`: Total variation on material maps
- `metallic_prior_loss()`: Encourages binary (0 or 1) metallic values
- `roughness_prior_loss()`: Soft regularization toward target roughness
- `albedo_chroma_loss()`: Consistency in color chromaticity
- `compute_pbr_losses()`: Combines all PBR regularizations

**Parameters**:
```python
compute_pbr_losses(
    gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
    alpha_map=None,
    lambda_albedo_smooth=0.01,
    lambda_roughness_smooth=0.01,
    lambda_metallic_smooth=0.01,
    lambda_metallic_prior=0.001,
    lambda_roughness_prior=0.001,
    lambda_albedo_chroma=0.001,
)
```

---

## 3. Key Classes and Their Interfaces

### 3.1 GaussianModel (Extended)

```python
class GaussianModel:
    def __init__(self, sh_degree: int, use_pbr: bool = False):
        """Initialize Gaussian model with optional PBR"""
        
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """Initialize Gaussians from point cloud"""
        # Sets: _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation
        # If use_pbr: also sets _albedo, _roughness, _metallic
        
    def training_setup(self, training_args):
        """Create optimizer with all learnable parameters"""
        # Base: xyz, f_dc, f_rest, opacity, scaling, rotation
        # PBR: albedo (lr: 0.001), roughness (lr: 0.0002), metallic (lr: 0.0002)
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """Adaptive point cloud densification"""
        
    # Properties with activation functions:
    @property
    def get_albedo(self) -> Tensor:     # [N, 3] in [0, 1]
    @property
    def get_roughness(self) -> Tensor:  # [N, 1] in [0.1, 0.999]
    @property
    def get_metallic(self) -> Tensor:   # [N, 1] in [0, 1]
```

### 3.2 Scene

```python
class Scene:
    def __init__(
        self, 
        args: ModelParams, 
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0]
    ):
        """Load scene from COLMAP or Blender dataset"""
        
    def getTrainCameras(self, scale=1.0) -> List[Camera]:
    def getTestCameras(self, scale=1.0) -> List[Camera]:
    def save(self, iteration: int):
        """Save Gaussians as PLY"""
```

### 3.3 EnvironmentLight

```python
class EnvironmentLight(nn.Module):
    def __init__(self, env_map_path: str = None, resolution: int = 512):
        """
        Args:
            env_map_path: Path to .hdr, .exr, or .png
            resolution: Height of environment map (width = 2*height for equirect)
        """
        
    def sample(self, directions: Tensor) -> Tensor:
        """
        Sample environment map given directions.
        Args:
            directions: [N, 3] or [H, W, 3] unit vectors
        Returns:
            colors: RGB values at those directions
        """
        
    def forward(self, directions: Tensor) -> Tensor:
        """Alias for sample()"""
```

### 3.4 Camera

```python
class Camera(nn.Module):
    def __init__(
        self,
        colmap_id, R, T, FoVx, FoVy,
        image, gt_alpha_mask,
        image_name, uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda"
    ):
        """Initialize camera with intrinsics/extrinsics"""
        
    # Properties:
    # .original_image: [3, H, W] or [4, H, W] (with alpha)
    # .image_height, .image_width
    # .world_view_transform: [4, 4] transformation matrix
    # .full_proj_transform: [4, 4] projection matrix
    # .camera_center: [3] world-space camera position
    # .FoVx, .FoVy: Field of view in X/Y
    # .znear, .zfar: Near/far planes
```

---

## 4. Training Pipeline (train_pbr.py)

### 4.1 High-Level Flow

```python
def training_pbr(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: List[int],
    checkpoint: str = None,
    env_map_path: str = None,
):
    """Main PBR training loop"""
```

### 4.2 Training Loop Overview

```python
# 1. Initialize
gaussians = GaussianModel(sh_degree=3, use_pbr=True)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)

# 2. Load/create environment light
env_light = EnvironmentLight(env_map_path, resolution=256).cuda()
env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=0.01)

# 3. Main iteration loop
for iteration in range(first_iter, opt.iterations + 1):
    # Pick random camera
    viewpoint_cam = scene.getTrainCameras().sample()
    
    # Render with G-Buffer
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_pbr=True)
    
    # Loss computation:
    # - Base: L1 + SSIM on SH rendering
    loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim)
    
    # - Normal regularization (after iter 7000)
    normal_loss = lambda_normal * normal_error.mean()
    
    # - Depth distortion (after iter 3000)
    dist_loss = lambda_dist * depth_distortion.mean()
    
    # - PBR shading loss (after iter 5000)
    if iteration > 5000 and use_pbr:
        gbuffer_albedo = render_pkg['gbuffer_albedo']
        gbuffer_roughness = render_pkg['gbuffer_roughness']
        gbuffer_metallic = render_pkg['gbuffer_metallic']
        gbuffer_normal = render_pkg['rend_normal']
        
        # Screen-space PBR shading
        shaded_image = screen_space_pbr_shading(
            gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
            gbuffer_normal, gbuffer_depth,
            camera_center, world_view_transform,
            env_light=env_light
        )
        
        pbr_loss = lambda_pbr * pbr_reconstruction_loss(shaded_image, gt_image)
        
        # PBR material regularization (after iter 10000)
        if iteration > 10000:
            pbr_losses = compute_pbr_losses(
                gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
                alpha_map, lambda_albedo_smooth=0.01, ...
            )
            pbr_reg_loss = lambda_pbr_reg * pbr_losses['total_pbr_reg']
    
    # Total loss
    total_loss = loss + dist_loss + normal_loss + pbr_loss + pbr_reg_loss
    
    # Backward pass
    total_loss.backward()
    
    # Optimizer step
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad()
    
    if iteration > 5000:
        env_light_optimizer.step()
        env_light_optimizer.zero_grad()
    
    # Densification (before iteration 15000)
    if iteration < opt.densify_until_iter:
        gaussians.add_densification_stats(viewspace_points, visibility_filter)
        if iteration % densification_interval == 0:
            gaussians.densify_and_prune(...)
```

### 4.3 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `iterations` | 30,000 | Total training iterations |
| `position_lr_init` | 0.00016 | Initial position learning rate |
| `albedo_lr` | 0.001 | Albedo learning rate |
| `roughness_lr` | 0.0002 | Roughness learning rate |
| `metallic_lr` | 0.0002 | Metallic learning rate |
| `lambda_pbr` | 0.1 | PBR shading reconstruction weight |
| `lambda_pbr_reg` | 0.01 | PBR regularization weight |
| `env_light_lr` | 0.01 | Environment light learning rate |

### 4.4 Training Stages

| Stage | Iteration | Feature |
|-------|-----------|---------|
| Warmup | 0-500 | Densification starts |
| Geometry | 500-5000 | Main SH rendering optimization |
| PBR Init | 5000-7000 | PBR shading introduced, environment light optimized |
| PBR Reg | 7000-10000 | Normal consistency enabled, PBR regularization starts |
| Refinement | 10000-15000 | Full PBR optimization with all losses |
| Final | 15000-30000 | Fine-tuning, no more densification |

---

## 5. Rendering Pipeline (render_pbr.py)

### 5.1 Main Rendering Function

```python
def render_set(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    env_light: EnvironmentLight,
    views: str,  # "train" or "test"
    out_dir: str,
    split_name: str,
) -> Scene:
    """Render a set of views and save outputs"""
```

### 5.2 Outputs Generated

For each view, saves:

```
<out_dir>/
├── renders/              # SH-based RGB rendering
├── gt/                   # Ground truth images
├── pbr_shaded/           # PBR-shaded output (env-lit)
├── albedo/               # Extracted albedo map
├── roughness/            # Extracted roughness map
├── metallic/             # Extracted metallic map
├── normal/               # Surface normal visualization
└── depth/                # Depth map (turbo colormap)
```

### 5.3 G-Buffer Contents

After rendering with `render_pbr=True`:

```python
render_pkg = {
    'render': [3, H, W],           # SH-based RGB
    'gbuffer_albedo': [3, H, W],   # Base color
    'gbuffer_roughness': [1, H, W],# Roughness
    'gbuffer_metallic': [1, H, W], # Metallic
    'rend_normal': [3, H, W],      # World-space normal
    'surf_depth': [1, H, W],       # Surface depth
    'rend_alpha': [1, H, W],       # Opacity/alpha
    'rend_dist': [1, H, W],        # Depth distortion
}
```

### 5.4 Metric Computation

```python
def compute_metrics(renders_dir, gt_dir) -> Dict[str, float]:
    """Compute PSNR, SSIM, LPIPS metrics"""
    # Computes over all PNG files in directory
    # Returns: {"PSNR": float, "SSIM": float, "LPIPS": float}
```

### 5.5 Usage Example

```bash
python render_pbr.py \
    -m <model_path> \
    -s <dataset_path> \
    --iteration -1 \
    --env_map <hdr_path> \
    --compute_metrics
```

---

## 6. Dependencies and CUDA Extensions

### 6.1 Environment (`environment.yml`)

```yaml
name: surfel_splatting
python: 3.12
pytorch: 2.9.0
pytorch-cuda: 13.0
torchvision: 0.24.0
torchaudio: 2.9.0

Key packages:
- pillow, imageio (image I/O)
- open3d (point cloud ops)
- trimesh, scikit-image (geometry)
- plyfile (PLY I/O)
- opencv-python (image processing)
- tqdm (progress bars)
- lpips (perceptual loss)
```

### 6.2 CUDA Extensions Required

**1. Differentiable Surfel Rasterization**
```
submodules/diff-surfel-rasterization/
├── cuda_rasterizer/
│   ├── forward.cu        # Forward rasterization
│   ├── backward.cu       # Gradient computation
│   └── rasterizer_impl.cu
├── rasterize_points.cu
├── ext.cpp
└── setup.py
```

Compiled as: `diff_surfel_rasterization._C` module

**2. Simple KNN**
```
submodules/simple-knn/
├── spatial.cu            # Spatial search
├── simple_knn.cu         # KNN implementation
├── ext.cpp
└── setup.py
```

Compiled as: `simple_knn._C` module

### 6.3 Installation

```bash
# Install CUDA extensions
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn

# Or via conda environment
conda env create --file environment.yml
conda activate surfel_splatting
```

---

## 7. Data Format Requirements

### 7.1 COLMAP Format

Expected directory structure:
```
<dataset>/
├── sparse/
│   ├── cameras.bin/txt      # Camera intrinsics
│   ├── images.bin/txt       # Camera extrinsics and image names
│   └── points3D.bin/txt     # 3D point cloud
└── images/
    ├── <image1>.jpg/png
    ├── <image2>.jpg/png
    └── ...
```

**Point Cloud Format**: Initialization from COLMAP's SfM point cloud
- Points: 3D locations
- Colors: RGB per point

### 7.2 Blender/NeRF Format

Expected files:
```
<dataset>/
├── transforms_train.json    # Training camera parameters
├── transforms_test.json     # Test camera parameters
└── <split>/                 # "train" or "test"
    ├── r_0.png
    ├── r_1.png
    └── ...
```

**JSON Format**:
```json
{
  "camera_angle_x": <float>,
  "frames": [
    {
      "file_path": "train/r_0",
      "rotation": 0.0,
      "transform_matrix": [[...], [...], [...], [...]]
    },
    ...
  ]
}
```

### 7.3 Point Cloud Format

Internal `BasicPointCloud` representation:
```python
class BasicPointCloud(NamedTuple):
    points: np.array    # [N, 3] float32 XYZ
    colors: np.array    # [N, 3] float32 RGB in [0, 1]
    normals: np.array   # [N, 3] float32 (not used for initialization)
```

### 7.4 Saved Model Format

**PLY File Format** (with PBR):
```python
# Standard 3DGS attributes
x, y, z                          # Position
nx, ny, nz                       # Placeholder normals (computed at render time)
f_dc_0, f_dc_1, f_dc_2          # SH DC coefficient [1, 3]
f_rest_0, ... f_rest_45         # SH higher orders [15, 3]
opacity                          # Single opacity value
scale_0, scale_1                # Scaling factors
rot_0, rot_1, rot_2, rot_3      # Quaternion rotation

# PBR attributes (when use_pbr=True)
albedo_0, albedo_1, albedo_2    # Base color RGB
roughness                        # Roughness [0.1, 0.999]
metallic                         # Metallic [0, 1]
```

---

## 8. Public APIs for Integration

### 8.1 Core Training API

```python
# Initialize model
gaussians = GaussianModel(sh_degree=3, use_pbr=True)

# Load data
scene = Scene(args, gaussians)

# Setup training
gaussians.training_setup(opt_args)
env_light = EnvironmentLight(env_map_path).cuda()

# Single training iteration
render_pkg = render(camera, gaussians, pipe, bg_color, render_pbr=True)

# Extract G-Buffer
albedo = render_pkg['gbuffer_albedo']
roughness = render_pkg['gbuffer_roughness']
metallic = render_pkg['gbuffer_metallic']
normal = render_pkg['rend_normal']

# PBR shading
shaded = screen_space_pbr_shading(
    albedo, roughness, metallic, normal, depth,
    camera_center, camera_transform, env_light
)

# Loss computation
pbr_loss = pbr_reconstruction_loss(shaded, gt_image)
pbr_regs = compute_pbr_losses(albedo, roughness, metallic)

# Optimizer step
total_loss.backward()
gaussians.optimizer.step()
```

### 8.2 Core Rendering API

```python
# Load trained model
gaussians = GaussianModel(sh_degree=3, use_pbr=True)
scene = Scene(args, gaussians, load_iteration=final_iter)

# Load environment light
env_light = EnvironmentLight().cuda()
env_light.load_state_dict(torch.load('env_light.pth'))

# Render with G-Buffer
render_pkg = render(camera, gaussians, pipe, bg_color, render_pbr=True)

# Extract and shade
albedo = render_pkg['gbuffer_albedo']
roughness = render_pkg['gbuffer_roughness']
metallic = render_pkg['gbuffer_metallic']
normal = render_pkg['rend_normal']
depth = render_pkg['surf_depth']

shaded = screen_space_pbr_shading(
    albedo, roughness, metallic, normal, depth,
    camera.camera_center, camera.world_view_transform,
    env_light=env_light
)

# Save outputs
save_image(shaded, 'output.png')
save_image(albedo, 'albedo.png')
```

### 8.3 PBR-Specific APIs

```python
# Environment lighting
env_light = EnvironmentLight('path/to/env.hdr', resolution=512)
color = env_light.sample(direction_vectors)  # [H, W, 3]

# PBR shading (single light)
rgb = pbr_shading(
    albedo, roughness, metallic, normal,
    view_dir, light_dir, light_color, ambient
)

# PBR shading (environment)
rgb = pbr_shading_env(albedo, roughness, metallic, normal, view_dir, env_light)

# Loss functions
recon_loss = pbr_reconstruction_loss(predicted, ground_truth)
material_loss = material_smoothness_loss(material_map, alpha)
metallic_loss = metallic_prior_loss(metallic_map)
```

### 8.4 Gaussian Model APIs

```python
# Initialize from point cloud
pcd = BasicPointCloud(points, colors, normals)
gaussians.create_from_pcd(pcd, spatial_lr_scale)

# Get material properties (with activations)
albedo = gaussians.get_albedo       # [N, 3] in [0, 1]
roughness = gaussians.get_roughness # [N, 1] in [0.1, 0.999]
metallic = gaussians.get_metallic   # [N, 1] in [0, 1]

# Save/load
gaussians.save_ply('model.ply')
gaussians.load_ply('model.ply')

# Densification
gaussians.densify_and_prune(max_grad, min_opacity, extent, screen_size)
```

---

## 9. Important Design Decisions

### 9.1 Material Parameterization

**Why Inverse Sigmoid**:
- Raw parameters unconstrained during optimization
- Sigmoid activation ensures valid ranges:
  - Albedo: (0, 1)
  - Roughness: (0.1, 0.999) clamped
  - Metallic: (0, 1)

**Initialization**:
- Albedo: From point cloud colors
- Roughness: 0.6 (moderately rough) - good for general materials
- Metallic: 0.12 (mostly dielectric) - most real-world objects

### 9.2 Two-Stage PBR Optimization

**Why delayed PBR loss (after 5000 iterations)**:
- Geometry must converge first
- PBR shading sensitivity to surface normal quality
- Prevents conflicting gradients early in training

**Why delayed regularization (after 10000 iterations)**:
- Material properties initially unconstrained
- Regularization prevents smooth unrealistic materials
- Allows proper material disentanglement

### 9.3 Deferred Rendering Choice

**G-Buffer Rasterization**:
- Renders each material property separately
- Enables flexible shading in post-process
- Compatible with standard PBR pipelines
- Allows real-time environment light changes

**vs. Direct Shading**:
- Could shade during forward rasterization
- Would reduce flexibility for integration
- G-Buffer enables material editing

### 9.4 Environment Light Design

**Learnable vs. Fixed**:
- Environment light is learnable to adapt to data
- Can load from HDR but parameters are optimized
- Balances realism with optimization flexibility

---

## 10. Key Limitations and Future Directions

### 10.1 Current Limitations

1. **View-Dependent Effects**: Basic normal-only; doesn't capture anisotropic BRDF
2. **Normal Accuracy**: Computed from depth; may have artifacts in thin regions
3. **Shadow/Occlusion**: No shadow computation; relies on environment lighting
4. **Material Complexity**: Limited to albedo/roughness/metallic; no normal maps, etc.
5. **Baking vs. Learning**: Material properties aren't position-dependent within Gaussians

### 10.2 Environment Map Geometric Representation Critique

**Issue**: The `EnvironmentLight` class in `utils/pbr_utils.py` uses **Equirectangular Projection** (LatLong Map) with a `[3, H, W]` tensor sampled via `grid_sample`. This parameterization has fundamental topological defects:

**1. Pole Singularity**
- At $\theta \to 0$ and $\theta \to \pi$ (poles), pixel density approaches infinity
- The optimizer allocates excessive parameters at poles while under-sampling the equatorial region (which typically contains primary lighting information)
- Mathematically: surface area element $dA = \sin\theta \, d\theta \, d\phi$, but pixel area is uniform in $(\theta, \phi)$ space

**2. Gradient Distortion**
- During backpropagation through `grid_sample`, the Jacobian determinant varies significantly across the sphere
- Gradients at poles exhibit unnecessary oscillation or overfitting due to geometric stretching
- The non-uniform sampling density causes SGD to converge unevenly across different solid angles

**3. Non-Conformal & Non-Area-Preserving**
- Equirectangular projection is neither conformal (angle-preserving) nor equal-area
- This introduces systematic bias in environment light optimization

**Recommended Improvements**:
| Representation | Pros | Cons |
|----------------|------|------|
| **CubeMap** | Uniform sampling per face, GPU-friendly | 6 discontinuities at edges |
| **HEALPix** | Equal-area pixels, optimal for spherical harmonics | Complex indexing, less GPU support |
| **Octahedral Map** | 2:1 aspect ratio, single texture | Minor distortion at corners |
| **Spherical Gaussians** | Compact, analytical integration | Limited high-frequency detail |

For academic rigor, CubeMap or Octahedral mapping would ensure uniform parameter density per unit solid angle, leading to more balanced SGD optimization.

### 10.2.1 Implemented Solution: Solid-Angle Weighted Optimization

To mitigate the equirectangular projection distortion without changing the representation, we implemented **solid-angle weighting** in `EnvironmentLight`:

**1. Weight Map Initialization**
```python
# Weight per pixel row: w(v) = sin(π·v)
v = torch.linspace(0, 1, H)
solid_angle_weight = torch.sin(np.pi * v).clamp(min=1e-6)
# Registered as buffer: self.solid_angle_weight [1, H, W]
```

**2. Weighted TV Regularization**
```python
# Standard TV loss over-penalizes poles; weighted version corrects this
env_tv_loss = lambda_env_tv * env_light.tv_loss_weighted()
```
The `tv_loss_weighted()` method scales gradient penalties by $\sin(\theta)$, ensuring uniform regularization per unit solid angle.

**3. Gradient Scaling Hook**
```python
# Reduces gradient magnitude at poles proportionally
env_light.register_gradient_scaling_hook()
# Hook: grad = grad * solid_angle_weight
```
This prevents pole pixels from dominating gradient updates during optimization.

**New Command-Line Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda_env_tv` | 0.001 | Solid-angle weighted TV loss weight |
| `--no_env_gradient_scaling` | False | Disable gradient scaling hook |

**Mathematical Justification**:
- Spherical surface element: $d\omega = \sin\theta \, d\theta \, d\phi$
- Equirectangular pixel area: $dA_{pixel} = \text{const}$ (uniform grid)
- Correction factor: $w(\theta) = \sin\theta$ normalizes contribution per steradian

### 10.3 Sampling Theory Critique: Roughness and Integration Approximation

**Issue**: The `sample()` function in `EnvironmentLight` performs **point sampling** at the reflection direction. This is a severe mathematical simplification of the rendering equation.

**The Rendering Equation Requires Convolution**:

The specular radiance should be computed as:
$$L_{spec} = \int_{\Omega} f_r(\omega_i, \omega_o) L_i(\omega_i) \cos\theta_i \, d\omega_i$$

where $f_r$ is the BRDF lobe. Point sampling approximates this as:
$$L_{spec} \approx L_i(\omega_{reflect})$$

**1. High-Frequency Aliasing**

- For **low roughness** (smooth/mirror-like surfaces): Point sampling is an acceptable approximation since the BRDF lobe is narrow (approaching a delta function).
- For **high roughness** (diffuse-like surfaces): The BRDF lobe spans a wide cone. Physically correct rendering requires sampling a region of the environment map, not a single point.

**2. Roughness Parameter Invalidation**

Without proper integration, the roughness parameter loses its geometric meaning:

| Expected Behavior | Actual Behavior |
|-------------------|-----------------|
| Higher roughness → blurrier reflections | Reflection sharpness unchanged |
| Roughness controls lobe width | Roughness only modulates intensity via Fresnel |
| Physically interpretable parameter | Becomes an arbitrary intensity scalar |

This causes **PBR physical interpretability collapse**: the roughness parameter can still be optimized to minimize reconstruction loss, but it no longer corresponds to actual surface micro-geometry.

**3. Mathematical Formulation of the Problem**

The GGX BRDF lobe has half-angle distribution:
$$D(\mathbf{h}) = \frac{\alpha^2}{\pi((\mathbf{n} \cdot \mathbf{h})^2(\alpha^2 - 1) + 1)^2}$$

where $\alpha = \text{roughness}^2$. For proper integration:
- $\alpha \to 0$: Lobe approaches delta function → point sampling valid
- $\alpha \to 1$: Lobe spreads over hemisphere → requires area sampling

**4. Standard Solutions (Not Implemented)**

| Technique | Description | Trade-off |
|-----------|-------------|-----------|
| **Split-Sum Approximation** | Pre-filtered environment mipmap + BRDF LUT | Requires precomputation, GPU-friendly |
| **Importance Sampling** | Monte Carlo with GGX-weighted samples | Runtime cost, variance/noise |
| **Spherical Gaussians** | Analytical lobe approximation | Limited accuracy for high roughness |
| **Pre-integrated BRDF** | Bake roughness into environment map levels | Memory overhead, less flexible |

**5. Implications for Optimization**

During training, the optimizer may:
- Push roughness toward extreme values (0 or 1) to compensate for missing blur
- Learn albedo that "bakes in" the missing specular convolution
- Produce materials that don't transfer correctly to other renderers

**Recommendation**: For physically accurate PBR, implement Split-Sum approximation:
```python
# Pseudocode for Split-Sum
def sample_prefiltered(direction, roughness):
    mip_level = roughness * max_mip_levels
    return env_map.sample_mip(direction, mip_level)

# Separate diffuse and specular integrals
diffuse = env_map.sample_irradiance(normal)
specular = sample_prefiltered(reflect_dir, roughness) * brdf_lut(NdotV, roughness)
```

### 10.3.1 Implemented Solution: Differentiable Prefiltered Mipmaps

We implemented the Split-Sum approximation with fully differentiable mipmap generation:

**1. Dynamic Mipmap Pyramid Construction**

```python
# In EnvironmentLight.__init__:
self.num_mip_levels = 5  # Level 0 = sharp, Level 4 = blurred

# Blur sigma increases exponentially per level
# sigma = 2^level (2, 4, 8, 16)
```

The mipmaps are built on-the-fly using separable Gaussian convolution, ensuring gradients flow back to the base environment map parameters.

**2. Trilinear Sampling with Roughness**

```python
def sample_prefiltered(self, directions, roughness):
    mipmaps = self._build_mipmaps()  # Differentiable

    # LOD from roughness: r=0 → level 0, r=1 → level N
    lod = roughness * (num_mip_levels - 1)
    lod_floor, lod_ceil = floor(lod), ceil(lod)

    # Trilinear interpolation
    val_floor = grid_sample(mipmaps[lod_floor], uv)
    val_ceil = grid_sample(mipmaps[lod_ceil], uv)
    return lerp(val_floor, val_ceil, frac(lod))
```

**3. Updated PBR Shading**

```python
def pbr_shading_env(albedo, roughness, metallic, normal, view_dir, env_light):
    # Specular: roughness-dependent blur
    specular_color = env_light.sample_prefiltered(reflect_dir, roughness)

    # Diffuse: maximum blur (irradiance approximation)
    diffuse_irradiance = env_light.sample_prefiltered(normal, roughness=1.0)

    # Fresnel with roughness correction
    F = fresnel_schlick_roughness(NdotV, F0, roughness)

    return kD * albedo * diffuse + kS * specular_color
```

**4. Physical Correctness Achieved**

| Roughness | Mip Level | Visual Result |
|-----------|-----------|---------------|
| 0.0 | 0 (sharp) | Mirror-like reflection |
| 0.25 | 1 | Slightly blurred |
| 0.5 | 2 | Moderately blurred |
| 0.75 | 3 | Heavily blurred |
| 1.0 | 4 (max blur) | Diffuse-like (no sharp features) |

**5. Gradient Flow**

The entire pipeline is differentiable:
- $\frac{\partial L}{\partial \text{roughness}}$ is now non-trivial (affects blur level)
- Optimizer learns physical roughness values that match observed reflection blur
- Materials transfer correctly to standard PBR renderers

**New API**:
```python
# Roughness-aware sampling
color = env_light.sample_prefiltered(directions, roughness)

# Legacy point sampling (for compatibility)
color = env_light.sample(directions)
```

### 10.4 Integration Considerations

- **Performance**: G-Buffer rasterization adds ~10-20% overhead
- **Memory**: Stores extra material parameters (~5% more VRAM)
- **Convergence**: Requires careful loss weight tuning
- **Initialization**: Material initialization critical for convergence

---

## 11. Example Workflows

### 11.1 Training Workflow

```python
# 1. Load COLMAP dataset
args = ModelParams()
args.source_path = "/path/to/colmap"
args.model_path = "./output/scene"

# 2. Initialize
gaussians = GaussianModel(3, use_pbr=True)
scene = Scene(args, gaussians)
opt = OptimizationParams()
gaussians.training_setup(opt)

# 3. Training loop
env_light = EnvironmentLight("env.hdr").cuda()
optimizer_env = torch.optim.Adam(env_light.parameters(), lr=0.01)

for iter in range(30000):
    # Random camera sampling
    cam = scene.getTrainCameras().sample()
    
    # Render
    render_pkg = render(cam, gaussians, pipe, bg, render_pbr=True)
    
    # Loss
    loss = compute_loss(render_pkg, cam.original_image, iter)
    
    # Backward
    loss.backward()
    gaussians.optimizer.step()
    if iter > 5000:
        optimizer_env.step()
```

### 11.2 Rendering Workflow

```python
# Load trained model
gaussians = GaussianModel(3, use_pbr=True)
scene = Scene(args, gaussians, load_iteration=30000)

# Load environment
env_light = EnvironmentLight().cuda()
env_light.load_state_dict(torch.load('env_light_30000.pth'))

# Render all test views
for cam in scene.getTestCameras():
    pkg = render(cam, gaussians, pipe, bg, render_pbr=True)
    
    # Extract G-Buffer
    albedo = pkg['gbuffer_albedo']
    roughness = pkg['gbuffer_roughness']
    metallic = pkg['gbuffer_metallic']
    normal = pkg['rend_normal']
    depth = pkg['surf_depth']
    
    # Shade with different environments
    shaded = screen_space_pbr_shading(
        albedo, roughness, metallic, normal, depth,
        cam.camera_center, cam.world_view_transform,
        env_light=env_light
    )
    
    # Save
    torchvision.utils.save_image(shaded, f'output/{cam.image_name}.png')
```

---

## 12. Integration Checklist

For integrating 2DGS-PBR into another project:

- [ ] Install CUDA extensions (diff-surfel-rasterization, simple-knn)
- [ ] Prepare dataset (COLMAP or Blender format)
- [ ] Configure arguments (ModelParams, OptimizationParams, PipelineParams)
- [ ] Initialize GaussianModel with use_pbr=True
- [ ] Create Scene and load cameras
- [ ] Setup EnvironmentLight with HDR map
- [ ] Implement training loop with PBR losses
- [ ] Extract G-Buffer for rendering
- [ ] Implement PBR shading pipeline
- [ ] Evaluate with metrics (PSNR, SSIM, LPIPS)
- [ ] Render material maps for visualization

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Core Model** | 2D Gaussians + PBR materials |
| **Learnable Materials** | Albedo (3D), Roughness (1D), Metallic (1D) per Gaussian |
| **Rendering** | Differentiable surfel rasterization + deferred PBR shading |
| **Lighting** | Learnable environment map (HDR) |
| **BRDF** | Cook-Torrance with Fresnel-Schlick |
| **Training Time** | ~30,000 iterations (~2-4 hours on A100) |
| **Material Init** | Albedo from colors; roughness 0.6; metallic 0.12 |
| **PBR Start** | Iter 5000 (after geometry convergence) |
| **Regularization** | Iter 10000+ (smoothness, priors, chroma) |
| **Output** | RGB + G-Buffer (albedo, roughness, metallic, normal, depth) |

