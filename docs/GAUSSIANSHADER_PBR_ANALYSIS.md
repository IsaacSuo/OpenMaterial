# GaussianShader PBR Implementation Analysis

## Executive Summary

GaussianShader implements PBR (Physically-Based Rendering) for 3D Gaussian Splatting by:
1. **Replacing SH color representation** with explicit material properties (albedo, roughness, metallic, normal)
2. **Using a learnable environment map** (cubemap-based) for dynamic lighting
3. **Implementing Split-Sum approximation** for IBL (Image-Based Lighting)
4. **Adding normal regularization** through pseudo-GT normal supervision

This document identifies key differences from your planned 2DGS PBR approach and highlights critical implementation details.

---

## 1. MODEL STRUCTURE: PBR PARAMETERS IN GAUSSIANMODEL

### File Location
`external/GaussianShader/scene/gaussian_model.py` (Lines 26-432)

### Key PBR Parameters

#### 1.1 Material Properties Storage

```python
# Lines 52-58
self.brdf_dim = brdf_dim                    # SH dimension for env map features
self.brdf_mode = brdf_mode                  # "envmap" mode
self.brdf_envmap_res = brdf_envmap_res      # Resolution of environment map (default 64)
self._normal = torch.empty(0)               # [N, 3] Primary normal
self._normal2 = torch.empty(0)              # [N, 3] Secondary normal (for view-facing selection)
self._specular = torch.empty(0)             # [N, 3] Specular color (F0)
self._roughness = torch.empty(0)            # [N, 1] Roughness value
```

#### 1.2 Key Difference #1: Dual Normal Representation
**GaussianShader stores TWO normals per Gaussian** (lines 55-56):
- `_normal` and `_normal2` are stored separately
- Selected based on view direction (see `get_normal()` at lines 109-123)
- This handles back-facing surfaces elegantly without flipping discontinuities

```python
def get_normal(self, dir_pp_normalized=None, return_delta=False):
    normal_axis = self.get_minimum_axis  # Covariance's smallest axis
    delta_normal1 = self._normal
    delta_normal2 = self._normal2
    delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1)
    # Select delta_normal based on view direction (positive or negative)
    idx = torch.where(positive, 0, 1).long()
    delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1)
    normal = delta_normal + normal_axis  # Offset from covariance axis
    normal = normal / normal.norm(dim=1, keepdim=True)
    return normal
```

**Your planned implementation**: Only uses single normal. Consider whether back-face handling is needed for 2DGS.

#### 1.3 Key Difference #2: Specular Color NOT Roughness/Metallic
**GaussianShader uses `_specular` (F0) directly** instead of metallic parameter:
- `_specular` represents the specular reflection (Fresnel F0)
- This is more flexible than metallic/roughness decomposition
- Allows arbitrary F0 without metallic constraint

**Your plan assumes**: metallic + roughness, which is standard metallic workflow.
**Trade-off**: GaussianShader's approach is more expressive but harder to regularize.

#### 1.4 Activation Functions

```python
# Lines 65-70
self.diffuse_activation = torch.sigmoid
self.specular_activation = torch.sigmoid
self.roughness_activation = torch.sigmoid
self.roughness_bias = 0.          # Can add bias to prevent too-smooth
self.default_roughness = 0.6      # Initialized to 0.6 (moderately rough)
```

### 1.5 Initialization from Point Cloud

**Lines 149-202** (in `create_from_pcd`):

```python
# Case: BRDF mode with envmap, brdf_dim > 0
fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
features = torch.zeros((fused_color.shape[0], 3)).float().cuda()  # Just RGB
features[:, :3] = fused_color  # Colors go into features_dc, NOT albedo!
features_rest = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2))

# Normals initialized to zero (trained from scratch!)
normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
self._normal = nn.Parameter(torch.from_numpy(normals).requires_grad_(True))

# Specular initialized to zeros
self._specular = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda"))

# Roughness initialized to default (0.6)
self._roughness = nn.Parameter(
    self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
)
```

**Key insight**: _features_dc stores the colored appearance (like diffuse), not albedo decomposition. The BRDF evaluation (lines 138-150 in render function) computes PBR shading on top.

---

## 2. RENDERING PIPELINE: PBR SHADING

### File Location
`external/GaussianShader/gaussian_renderer/__init__.py` (Lines 62-268)

### 2.1 High-Level Flow

```
render() [Line 62]
  ├─ For BRDF mode (line 126-150):
  │  ├─ Extract PBR parameters:
  │  │  - normal from _normal/_normal2 (line 134)
  │  │  - specular F0 (line 136)
  │  │  - roughness (line 137)
  │  │  - diffuse color from features_dc (line 133)
  │  │
  │  ├─ Call brdf_mlp.shade() [Line 138]
  │  │  └─ Computes PBR shading with environment map
  │  │
  │  └─ Optional: Add SH residual for features_rest (lines 144-150)
  │
  ├─ Rasterize to image [Line 167]
  └─ Post-processing:
     ├─ Render G-buffers (normal, depth, roughness, etc.) [Lines 177-220]
     └─ Compute pseudo-GT normal from depth [Line 250]
```

### 2.2 PBR Shading Call

**Lines 138-150** show the core shading:

```python
gb_pos = pc.get_xyz                  # [N, 3] world positions
view_pos = viewpoint_camera.camera_center.repeat(N, 1)  # [N, 3] camera position
diffuse = pc.get_diffuse             # [N, 3] = _features_dc
normal = pc.get_normal(dir_pp_normalized)  # [N, 3]
specular = pc.get_specular           # [N, 3] F0
roughness = pc.get_roughness         # [N, 1]

# CORE: Call environment map shading
color, brdf_pkg = pc.brdf_mlp.shade(
    gb_pos[None, None, ...],          # [1, 1, N, 3]
    normal[None, None, ...],          # [1, 1, N, 3]
    diffuse[None, None, ...],         # [1, 1, N, 3]
    specular[None, None, ...],        # [1, 1, N, 3]
    roughness[None, None, ...],       # [1, 1, N, 1]
    view_pos[None, None, ...]         # [1, 1, N, 3]
)
colors_precomp = color.squeeze()      # [N, 3]
```

### 2.3 Environment Map Shading (EnvironmentLight.shade)

**File**: `external/GaussianShader/scene/NVDIFFREC/light.py` (Lines 128-174)

```python
def shade(self, gb_pos, gb_normal, kd, ks, kr, view_pos, specular=True):
    # kd: diffuse color (N, 3)
    # ks: specular F0 (N, 3)
    # kr: roughness (N, 1)
    
    wo = safe_normalize(view_pos - gb_pos)  # View direction
    
    # Step 1: Diffuse from environment
    nrmvec = gb_normal
    ambient = dr.texture(
        self.diffuse[None, ...],  # Pre-computed diffuse irradiance cubemap
        nrmvec.contiguous(),
        filter_mode='linear',
        boundary_mode='cube'
    )
    specular_linear = ambient * (1.0 - ks)  # Diffuse modulated by (1-F0)
    
    # Step 2: Specular from environment (with BRDF LUT)
    NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
    fg_uv = torch.cat((NdotV, kr), dim=-1)  # (N-dot-V, roughness) lookup
    fg_lookup = dr.texture(
        self._FG_LUT,  # Pre-loaded BRDF LUT (256x256x2)
        fg_uv,
        filter_mode='linear'
    )  # Returns [F, (1-F)]
    
    # Step 3: Specular environment lookup with mips
    reflvec = safe_normalize(reflect(wo, gb_normal))
    miplevel = self.get_mip(roughness)  # Map roughness to mip level
    spec = dr.texture(
        self.specular[0][None, ...],  # Base cubemap
        reflvec.contiguous(),
        mip=self.specular[1:],  # Mipped versions
        mip_level_bias=miplevel[..., 0],
        filter_mode='linear-mipmap-linear'
    )
    
    # Step 4: Combine using Fresnel split-sum
    reflectance = ks * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
    specular_linear += spec * reflectance
    
    # Step 5: Diffuse contribution
    diffuse_linear = torch.sigmoid(kd - np.log(3.0))
    
    rgb = specular_linear + diffuse_linear
    return rgb, extras
```

**Key concepts**:
1. **Split-Sum Approximation**: Separates diffuse and specular IBL
2. **BRDF LUT**: Pre-computed lookup table for (NdotV, roughness) → (Fresnel, kD)
3. **Mipmap progression**: Rougher surfaces use more blurred mip levels
4. **Diffuse irradiance**: Pre-computed from lowest mip (fully convolved)

### 2.4 G-Buffer Rendering

**Lines 177-220** render additional feature maps:

```python
render_extras = {
    "depth": depth,
    "normal": normal,                    # [3, H, W]
    "delta_normal_norm": delta_normal_norm,  # [3, H, W]
    "alpha": alpha,
    "diffuse": diffuse,
    "specular": specular,
    "roughness": roughness,
}

for k in render_extras.keys():
    out_extras[k] = rasterizer(
        colors_precomp=render_extras[k],
        # ... other params same
    )[0]
```

Then computes pseudo-GT normal from depth (line 250):
```python
out_extras["normal_ref"] = render_normal(
    depth=out_extras['depth'][0],
    ...
)  # Sobel or similar to compute normal from depth gradient
```

---

## 3. TRAINING: LOSS FUNCTIONS

### File Locations
- Loss definitions: `external/GaussianShader/utils/loss_utils.py` (Lines 71-119)
- Training loop: `external/GaussianShader/train.py` (Lines 76-99)

### 3.1 Loss Components

#### 3.1.1 RGB Reconstruction Loss

```python
# Line 96-97
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
```

#### 3.1.2 Normal Supervision (Pseudo-GT from Depth)

**Lines 86-88**:
```python
if iteration > opt.normal_reg_from_iter and iteration < opt.normal_reg_util_iter:
    losses_extra['predicted_normal'] = predicted_normal_loss(
        render_pkg["normal"],      # Rendered normal
        render_pkg["normal_ref"],  # GT normal (computed from depth)
        render_pkg["alpha"]
    )
```

**Function** (`loss_utils.py` lines 77-98):
```python
def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss (from ref-NeRF)"""
    # Erodes alpha to remove boundary artifacts
    weight = erode(alpha[0], erode_size=4)
    
    # Dot product loss: maximize alignment
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()
    # Minimizes (1 - cos_similarity), i.e., maximizes alignment
    
    return loss
```

**Key insight**: Uses rendered depth → normal as proxy supervision. This avoids needing external normal estimates.

#### 3.1.3 Alpha (Opacity) Regularization

**Line 89**:
```python
losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])
```

**Function** (`loss_utils.py` lines 71-75):
```python
def zero_one_loss(img):
    """Encourages alpha to be either 0 or 1 (binary)"""
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    # Minimizes log(x) + log(1-x), which is minimized at x=0 or x=1
    return loss
```

#### 3.1.4 Delta Normal Regularization

**Lines 91-92**:
```python
if "delta_normal_norm" in render_pkg.keys() and opt.lambda_delta_reg > 0:
    losses_extra['delta_reg'] = delta_normal_loss(
        render_pkg["delta_normal_norm"],
        render_pkg["alpha"]
    )
```

**Function** (`loss_utils.py` lines 100-119):
```python
def delta_normal_loss(delta_normal_norm, alpha=None):
    """Encourages minimal deviation of per-point normal from covariance axis"""
    weight = erode(alpha[0], erode_size=4)
    w = weight[..., 0].detach()
    l = delta_normal_norm[..., 0]  # Norm of deviation
    loss = (w * l).mean()
    # Minimizes the magnitude of delta_normal
    return loss
```

**Purpose**: Keeps normals close to Gaussian's principal axis (prevents wild variations).

### 3.2 Total Loss

**Line 98-99**:
```python
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
for k in losses_extra.keys():
    loss += getattr(opt, f'lambda_{k}') * losses_extra[k]
```

**Default weights** (`arguments/__init__.py` lines 111-113):
```python
self.lambda_zero_one = 1e-3
self.lambda_predicted_normal = 2e-1      # 20% weight!
self.lambda_delta_reg = 1e-3
```

### 3.3 Training Schedule for Normals

**Lines 77-78**:
```python
gaussians.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
gaussians.set_requires_grad("normal2", state=iteration >= opt.normal_reg_from_iter)
```

**Rationale**: Only optimize normals after RGB stabilizes (default starts at iter 0).

---

## 4. KEY DIFFERENCES FROM YOUR PLANNED APPROACH

### Table: GaussianShader vs Your 2DGS Plan

| Aspect | GaussianShader | Your Plan | Trade-off |
|--------|---|---|---|
| **Diffuse representation** | RGB in _features_dc | Separate albedo parameter | GS: simpler, can capture non-PBR effects. Your: cleaner decomposition |
| **Specular representation** | Direct F0 (3 channels) | Metallic + F0 (0.04 constraint) | GS: more flexible. Your: more interpretable |
| **Normal storage** | Dual normals + delta | Single normal | GS: handles back-faces smoothly. Your: simpler but risks discontinuities |
| **Environment map** | Learnable cubemap (6x64x64) | Learnable cubemap or SH | GS: detailed. Your: can use SH for memory efficiency |
| **Roughness initialization** | 0.6 (moderately rough) | 0.5 (sigmoid(0)) | GS: avoids too-shiny default |
| **Normal initialization** | Zeros (trained from scratch) | Could initialize from SfM or covariance axis | GS: cleaner learning problem. Your: could improve convergence |
| **Normal supervision** | Pseudo-GT from depth | Plan mentions external (Marigold) or pseudo-GT | GS: self-supervised. Your: more flexible |
| **BRDF LUT** | Pre-computed NVIDIA file | Would need to implement or load | GS: avoids optimization cost. Your: could integrate into network |
| **SH residual** | Optional (features_rest) | Plan mentions SH as fallback | GS: uses for high-frequency details |

### 4.1 Critical Differences to Address

#### Difference #1: Dual Normal System
GaussianShader's `_normal` and `_normal2` system elegantly handles view-dependent normal selection:
```python
idx = torch.where(positive, 0, 1).long()  # Choose based on dot(view_dir, covariance_normal)
delta_normal = torch.gather(...)[idx]
```

**For 2DGS**, consider:
- Do you need this? 2D gaussians are typically for foreground objects
- If only single-sided, simple normal suffices
- If two-sided needed, implement this elegant solution

#### Difference #2: Using Rendered Depth for Normal Supervision
Instead of external normal predictor:
```python
# Compute depth during rendering
p_view = world_view_matrix @ [pos, 1]
depth = p_view.z

# Then compute normal from depth gradients
normal_ref = compute_normal_from_depth(depth)
```

**Advantage**: Fully self-supervised, no external dependency
**Your plan mentions**: Marigold for pseudo-GT (external model)
**Recommendation**: Use depth-based as primary, Marigold as optional validation

#### Difference #3: Normal Starts at Zero
GaussianShader initializes normals to zero and learns purely from supervision:
```python
normals = np.zeros_like(pcd.points)
self._normal = nn.Parameter(torch.from_numpy(normals).requires_grad_(True))
```

**Your plan might initialize from**: SfM covariance axis or point normals
**Trade-off**:
- Zero init: cleaner learning, but slower convergence
- Covariance init: faster but may bias toward geometry

#### Difference #4: Diffuse NOT Separated from Diffuse Irradiance
Key insight in lines 139-150 of render:
```python
diffuse = pc.get_diffuse  # [N, 3] = _features_dc
# ... shading ...
# The BRDF shader COMBINES this with environment irradiance
# It's not pure "albedo" - it's learned appearance
```

**Your plan assumes**: `albedo = get_diffuse * irradiance`
**GaussianShader does**: `appearance = learned_features_dc + PBR(env_map)`

This is subtle but important: GaussianShader's `_features_dc` can capture:
- View-dependent effects not explained by PBR
- Non-Lambertian diffuse (e.g., fabric anisotropy)
- Subsurface scattering

---

## 5. ENVIRONMENT MAP IMPLEMENTATION

### File: `scene/NVDIFFREC/light.py` (EnvironmentLight class)

### 5.1 Representation
```python
class EnvironmentLight(nn.Module):
    def __init__(self, base):
        self.base = nn.Parameter(base)  # [6, res, res, 3] cubemap
        
    def build_mips(self):
        self.specular = [self.base]
        # Progressively downsampled versions
        while self.specular[-1].shape[1] > 16:
            self.specular.append(cubemap_mip.apply(self.specular[-1]))
        
        # Compute diffuse irradiance (fully convolved)
        self.diffuse = diffuse_cubemap(self.specular[-1])
```

### 5.2 Roughness to Mip Mapping
```python
def get_mip(self, roughness):
    return torch.where(
        roughness < 0.5,
        (roughness - 0.08) / (0.5 - 0.08) * (len(self.specular) - 2),
        (roughness - 0.5) / (1.0 - 0.5) + len(self.specular) - 2
    )
```

Maps roughness linearly to mip levels, with two segments for better distribution.

### 5.3 Initialization
**Lines 206-208**:
```python
def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32) * scale + bias
    # Uniform random in [bias, bias+scale]
    # Default: [0.25, 0.75] (mid gray)
    return EnvironmentLight(base)
```

Initialized to mid-gray with noise (avoids black/white extremes).

---

## 6. LEARNING RATE CONFIGURATION

**File**: `arguments/__init__.py` (Lines 102-114)

```python
self.brdf_mlp_lr_init = 1.6e-2          # 0.016 - quite high!
self.brdf_mlp_lr_final = 1.6e-3          # 1.6e-3
self.brdf_mlp_lr_delay_mult = 0.01
self.brdf_mlp_lr_max_steps = 30_000

self.normal_lr = 0.0002
self.specular_lr = 0.0002
self.roughness_lr = 0.0002
```

**Observation**: Environment map has ~100x higher LR than material properties!

**Reason**: Environment map is global, affects all pixels. Materials are local, need careful tuning.

---

## 7. PSEUDO-GT NORMAL CREATION

**Key function**: `render_normal` in `gaussian_renderer/__init__.py` (Lines 35-45)

```python
def render_normal(viewpoint_cam, depth, bg_color, alpha):
    """Compute normal from rendered depth using camera calibration"""
    intrinsic_matrix = viewpoint_cam.K      # [3, 3]
    extrinsic_matrix = viewpoint_cam.E      # [4, 4]
    
    # Transform depth to point positions in world space
    normal_ref = normal_from_depth_image(
        depth,
        intrinsic_matrix,
        extrinsic_matrix
    )
    
    # Alpha composite with background
    background = bg_color[None, None, ...]
    normal_ref = normal_ref * alpha[..., None] + background * (1 - alpha[..., None])
    
    return normal_ref
```

**Implementation**: Likely uses Sobel filters on depth map to compute gradients, then cross product for normal.

---

## 8. NORMAL REGULARIZATION THROUGH DELTA

**Key innovation**: Constrains per-point normal variation

```python
# Covariance's smallest axis (normal_axis)
normal_axis = get_minimum_axis(scaling, rotation)

# Per-point learned delta
delta_normal = _normal  # Learned parameter

# Final normal
normal = normal_axis + delta_normal
normal = normalize(normal)
```

**Effect**: 
- Normals stay close to geometry (covariance axis)
- But can deviate for material details
- Regularized by `lambda_delta_reg` loss

This prevents normals from becoming completely disconnected from geometry.

---

## 9. IMPLEMENTATION CHECKLIST FOR 2DGS PBR

Based on GaussianShader analysis, here's what to prioritize:

### Phase 1: Model Structure
- [x] Add `_normal` parameter
- [ ] Consider: Do you need `_normal2`? (probably not for 2D foreground)
- [x] Add `_specular` (or decompose into metallic + roughness as planned)
- [x] Add `_roughness`
- [ ] Decision: Keep `_features_dc` as appearance, or separate into albedo?

### Phase 2: Rendering
- [ ] Output normal map (you already have from 2DGS)
- [ ] Output specular/roughness maps
- [ ] Render depth for pseudo-GT normal computation
- [ ] Implement `render_normal()` - compute normal from depth gradients

### Phase 3: PBR Shading
- [ ] **Option A (Simpler)**: Use GaussianShader's EnvironmentLight directly (if compatible with 2DGS rasterizer)
- [ ] **Option B (Clean)**: Implement minimal Split-Sum in PyTorch:
  - Diffuse: Environment irradiance lookup
  - Specular: Roughness-weighted mip sampling + BRDF LUT lookup
- [ ] Decide: Use SH for env light (simpler, lower memory) vs cubemap (higher fidelity)?

### Phase 4: Loss Functions
- [ ] RGB loss (you have)
- [ ] Normal supervision loss (use depth-based pseudo-GT, GaussianShader style)
- [ ] Alpha regularization (zero_one_loss)
- [ ] Delta normal regularization (if using delta from covariance)
- [ ] Optional: Material smoothness losses

### Phase 5: Training
- [ ] Learning rate scheduling (env map needs higher LR than materials)
- [ ] Normal gradient control (disable initially, enable after iter N)
- [ ] Densification for material parameters

---

## 10. CODE SNIPPETS FOR REFERENCE

### 10.1 Initialization Pattern

```python
# From GaussianShader gaussian_model.py lines 191-198
normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
self._normal = nn.Parameter(torch.from_numpy(normals).to(self._xyz.device).requires_grad_(True))

specular_len = 3
self._specular = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], specular_len), device="cuda"))

self._roughness = nn.Parameter(
    self.default_roughness * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
)
self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))
```

### 10.2 Optimizer Setup Pattern

```python
# From gaussian_model.py lines 221-226
if self.brdf:
    l.extend([
        {'params': list(self.brdf_mlp.parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
        {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
        {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
        {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
    ])
```

### 10.3 Rendering Pattern

```python
# From gaussian_renderer/__init__.py lines 130-150
diffuse = pc.get_diffuse
normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True)
specular = pc.get_specular
roughness = pc.get_roughness

color, brdf_pkg = pc.brdf_mlp.shade(
    gb_pos[None, None, ...],
    normal[None, None, ...],
    diffuse[None, None, ...],
    specular[None, None, ...],
    roughness[None, None, ...],
    view_pos[None, None, ...]
)
colors_precomp = color.squeeze()
```

### 10.4 Loss Pattern

```python
# From train.py lines 86-92
losses_extra = {}
if pipe.brdf and iteration > opt.normal_reg_from_iter:
    if iteration < opt.normal_reg_util_iter:
        losses_extra['predicted_normal'] = predicted_normal_loss(
            render_pkg["normal"], render_pkg["normal_ref"], render_pkg["alpha"]
        )
    losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])
    if "delta_normal_norm" in render_pkg.keys() and opt.lambda_delta_reg > 0:
        losses_extra['delta_reg'] = delta_normal_loss(
            render_pkg["delta_normal_norm"], render_pkg["alpha"]
        )
```

---

## 11. POTENTIAL INTEGRATION POINTS

### 11.1 Compatibility with 2DGS Rasterizer
- GaussianShader is built on 3DGS rasterizer
- 2DGS uses different rasterizer (2D projection)
- **Question**: Can you reuse EnvironmentLight.shade() directly?
  - Probably yes, it's just BRDF computation
  - But may need to adapt input shapes

### 11.2 BRDF LUT Dependency
- GaussianShader loads pre-computed BRDF LUT from file:
  ```python
  self._FG_LUT = torch.tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin'))
  ```
- This is a ~256x256x2 float32 lookup table
- **Your options**:
  1. Use same pre-computed file (simplest)
  2. Implement your own LUT computation
  3. Replace LUT with analytical formula (slower but no dependencies)

### 11.3 nvdiffrast Dependency
- EnvironmentLight uses nvdiffrast for cubemap texture lookups
- This is an NVIDIA library for differentiable rendering
- Already a dependency of GaussianShader
- Check if 2DGS project already has it

---

## 12. EXPERIMENTAL INSIGHTS

### 12.1 Why This Architecture Works

1. **Learnable Environment Map**: Captures global lighting without manual specification
2. **Per-Gaussian Normals**: Natural surface detail without explicit normal maps
3. **Depth-Based Pseudo-GT**: Self-supervised normal learning, avoids external models
4. **Split-Sum IBL**: Efficient approximation of complex lighting integral
5. **Dual Normal System**: Handles view-dependent effects smoothly

### 12.2 What to Watch Out For

1. **Normal Initialization**: Starting from zero works, but may converge slowly
2. **Learning Rate Tuning**: Environment map needs 100x higher LR than materials
3. **Boundary Artifacts**: Alpha erosion (size=4) removes edge discontinuities
4. **BRDF LUT Bottleneck**: If no BRDF LUT available, analytical formulas are needed
5. **Back-Face Handling**: Dual normal system is elegant but more complex than single normal

---

## References

**GaussianShader GitHub**: https://github.com/Asparagus15/GaussianShader
**Key Files**:
- `scene/gaussian_model.py`: Model definition (Lines 26-623)
- `gaussian_renderer/__init__.py`: Rendering pipeline (Lines 62-268)
- `scene/NVDIFFREC/light.py`: Environment map (Lines 44-174)
- `utils/loss_utils.py`: Loss functions (Lines 71-119)
- `arguments/__init__.py`: Hyperparameters (Lines 47-115)
