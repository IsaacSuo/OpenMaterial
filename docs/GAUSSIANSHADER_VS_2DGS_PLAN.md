# GaussianShader vs 2DGS PBR Plan: Detailed Comparison

## Quick Reference Matrix

| Feature | GaussianShader | 2DGS Plan | Recommendation |
|---------|---|---|---|
| **Model Parameters** | | | |
| Normals | Dual (_normal, _normal2) | Single | Use single for 2D-only |
| Specular | Direct F0 (3ch) | Metallic+F0 decomposition | Your decomposition is cleaner |
| Roughness | Per-pixel (1ch) | Per-pixel (1ch) | Same approach |
| Diffuse | Via _features_dc | Separate albedo | Separate is cleaner |
| **Rendering** | | | |
| Shading approach | Split-Sum IBL + BRDF LUT | Planned: similar | Align on implementation |
| Env map | Learnable cubemap (6x64) | Planned: cubemap or SH | Cubemap better quality |
| Normal prediction | Pseudo-GT from depth | Planned: Marigold or depth | Use depth (self-supervised) |
| **Training** | | | |
| Env LR | 1.6e-2 (high) | Not yet specified | Use high LR for env map |
| Material LR | 2e-4 (low) | Not yet specified | Use low LR for materials |
| Normal init | Zeros | Not specified | Zero works fine |
| Loss functions | RGB + normal + alpha reg | Planned | See details below |
| **Infrastructure** | | | |
| BRDF LUT | Pre-computed file | Needs implementation | Can reuse GS file or make analytical |
| Dependencies | nvdiffrast | Probably available | Check 2DGS setup |

---

## Detailed Analysis: What to Keep, What to Change

### 1. MODEL STRUCTURE (gaussian_model.py)

#### 1.1 What GaussianShader Does

```python
# In __init__:
self._normal = torch.empty(0)       # [N, 3] primary
self._normal2 = torch.empty(0)      # [N, 3] secondary (view-dependent)
self._specular = torch.empty(0)     # [N, 3] F0 directly
self._roughness = torch.empty(0)    # [N, 1]

# In get_normal():
# Selects between _normal and _normal2 based on view direction
# Final normal = normal_axis (from covariance) + selected_delta
```

#### 1.2 What Your Plan Proposes

```python
# From your doc:
self._albedo = torch.empty(0)       # [N, 3] base color
self._roughness = torch.empty(0)    # [N, 1]
self._metallic = torch.empty(0)     # [N, 1]
# (normal handling not fully specified)
```

#### 1.3 Recommended Hybrid Approach

```python
# Combine best of both:
self._albedo = torch.empty(0)           # [N, 3] - cleaner than _features_dc
self._roughness = torch.empty(0)        # [N, 1]
self._metallic = torch.empty(0)         # [N, 1] - more interpretable than _specular
self._normal = torch.empty(0)           # [N, 3] - single normal (sufficient for 2D)

# Optionally:
self._features_rest = torch.empty(0)    # [N, 3, (sh_dim+1)^2] - for high-freq residuals
```

**Why this works**:
- Single normal simpler than dual (2D objects mostly single-sided)
- Metallic decomposition more interpretable than direct F0
- Albedo separate from appearance (cleaner PBR semantics)
- Keep SH residuals for non-PBR details

---

### 2. RENDERING PIPELINE (gaussian_renderer/__init__.py)

#### 2.1 GaussianShader Approach

```
render():
  - Extract: normal, specular, roughness, diffuse
  - Call: brdf_mlp.shade(pos, normal, diffuse, specular, roughness, view_pos)
    - Internally uses Split-Sum with mipped cubemap
    - Returns: shaded_color, {diffuse_contrib, specular_contrib}
  - Rasterize colored gaussians
  - Post-process: render G-buffers (normal, depth, roughness, etc.)
  - Compute pseudo-GT normal from rendered depth
```

#### 2.2 Your Planned Approach

```
render():
  - Render G-buffer (albedo, normal, roughness, metallic, depth)
  - Compute shading in screen-space AFTER rasterization
  - Optional: add environment lighting
```

#### 2.3 Recommended Integration

**GaussianShader's approach is better for real-time quality:**

```python
# Pre-shading (GaussianShader style):
def render(viewpoint_camera, pc, pipe, bg_color, ...):
    # 1. Compute per-gaussian shading
    gb_pos = pc.get_xyz
    normal = pc.get_normal()
    albedo = pc.get_albedo()
    metallic = pc.get_metallic()
    roughness = pc.get_roughness()
    
    # Compute F0 from metallic
    F0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)
    
    # Shade with environment
    color, brdf_extras = shade_pbr(
        positions=gb_pos,
        normals=normal,
        albedo=albedo,
        F0=F0,
        roughness=roughness,
        view_pos=camera_pos,
        env_map=pc.env_mlp
    )
    
    # 2. Rasterize pre-shaded colors
    rendered_image, radii = rasterizer(
        means3D=gb_pos,
        colors_precomp=color,  # Already shaded!
        ...
    )
    
    # 3. Also render G-buffers for analysis
    # (optional, for debugging/relighting)
```

**Advantage**: Shading quality matches per-gaussian resolution (better than screen-space)

---

### 3. PBR SHADING IMPLEMENTATION

#### 3.1 GaussianShader (Split-Sum with BRDF LUT)

```python
def shade(gb_pos, gb_normal, kd, ks, kr, view_pos):
    # Step 1: Diffuse from environment
    ambient = texture(diffuse_cubemap, gb_normal)
    spec_out = ambient * (1 - ks)
    
    # Step 2: Specular BRDF lookup
    NdotV = dot(view_dir, gb_normal)
    F_lookup = texture(BRDF_LUT, [NdotV, kr])  # [F, (1-F)]
    
    # Step 3: Environment specular
    reflect_dir = reflect(view_dir, gb_normal)
    miplevel = roughness_to_mip(kr)
    spec_env = texture_mip(specular_cubemap, reflect_dir, miplevel)
    
    # Step 4: Combine
    spec_out += spec_env * (ks * F[0] + F[1])
    
    # Step 5: Diffuse contribution
    diffuse_out = sigmoid(kd)
    
    return spec_out + diffuse_out
```

**Requires**:
- Pre-computed diffuse irradiance (convolution of env map)
- Pre-computed BRDF LUT (256x256x2 float32)
- Mipped specular cubemaps

#### 3.2 Analytical Alternative (No BRDF LUT)

For 2DGS, if you want to avoid BRDF LUT file dependency:

```python
def shade_pbr_analytical(
    positions, normals, albedo, F0, roughness,
    view_pos, env_light_dir, env_light_color
):
    """
    Simple PBR with analytical BRDF (no LUT)
    Assumes single dominant light direction
    """
    V = normalize(view_pos - positions)  # View vector
    L = normalize(env_light_dir - positions)  # Light vector
    H = normalize(V + L)
    
    NdotL = clamp(dot(N, L))
    NdotV = clamp(dot(N, V))
    NdotH = clamp(dot(N, H))
    VdotH = clamp(dot(V, H))
    
    # Fresnel Schlick
    F = F0 + (1 - F0) * pow(1 - VdotH, 5)
    
    # Distribution GGX
    alpha = roughness * roughness
    denom = NdotH*NdotH * (alpha*alpha - 1) + 1
    D = (alpha*alpha) / (pi * denom * denom)
    
    # Geometry Smith
    ggx_v = NdotV / (NdotV * (1 - k) + k)
    ggx_l = NdotL / (NdotL * (1 - k) + k)
    G = ggx_v * ggx_l
    
    # Specular BRDF
    specular = D * F * G / (4 * NdotV * NdotL)
    
    # Diffuse Lambert
    kd = (1 - F) * (1 - metallic)
    diffuse = kd * albedo / pi
    
    # Final
    color = (diffuse + specular) * env_light_color * NdotL
    
    return color
```

**Pros**: No external dependencies
**Cons**: Less accurate (single light assumption), slower per-sample computation

---

### 4. LOSS FUNCTIONS: Critical Implementation Details

#### 4.1 What GaussianShader Uses

**Lines 86-99 in train.py**:

```python
# RGB loss (standard)
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

# Normal supervision (from pseudo-GT depth)
if iteration > 0:  # Start immediately
    losses_extra['predicted_normal'] = predicted_normal_loss(
        render_pkg["normal"],      # Rendered normal
        render_pkg["normal_ref"],  # Computed from depth
        render_pkg["alpha"]
    )

# Alpha regularization (encourage opacity binary)
losses_extra['zero_one'] = zero_one_loss(render_pkg["alpha"])

# Delta normal regularization (keep normals close to geometry)
if "delta_normal_norm" in render_pkg:
    losses_extra['delta_reg'] = delta_normal_loss(
        render_pkg["delta_normal_norm"],
        render_pkg["alpha"]
    )

# Combine with weights
for k in losses_extra:
    loss += getattr(opt, f'lambda_{k}') * losses_extra[k]
```

**Default weights**:
- lambda_dssim: 0.2
- lambda_predicted_normal: 0.2 (20%!)
- lambda_zero_one: 1e-3
- lambda_delta_reg: 1e-3

#### 4.2 Your Planned Approach (from your doc)

```python
# RGB loss
L_rgb = l1_loss + 0.1 * ssim_loss

# Normal supervision (mentions Marigold)
if normal_pseudo_gt:
    L_normal = l1_loss(rendered_normal, pseudo_gt)

# Material regularization
L_reg = 0.01 * tv_loss(roughness) + 0.01 * metallic.mean()

# Total
loss = L_rgb + 0.1 * L_normal + 0.05 * L_depth + L_reg
```

#### 4.3 Recommended Merge

```python
def compute_training_loss(
    render_pkg, gt_image, opt,
    use_pseudo_normal=True,
    use_marigold_normal=False,
    marigold_normal=None
):
    # 1. RGB loss (standard)
    ll1 = l1_loss(render_pkg['image'], gt_image)
    l_rgb = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(...))
    
    # 2. Normal supervision (primary: depth-based, secondary: external)
    l_normal = 0
    if use_pseudo_normal and "normal_ref" in render_pkg:
        l_normal += predicted_normal_loss(
            render_pkg["normal"],
            render_pkg["normal_ref"],  # Computed from depth
            render_pkg["alpha"]
        )
    
    if use_marigold_normal and marigold_normal is not None:
        # Optional: compare with external estimate
        l_normal += 0.5 * predicted_normal_loss(
            render_pkg["normal"],
            marigold_normal,
            render_pkg["alpha"]
        )
    
    # 3. Opacity regularization (GaussianShader's clever trick)
    l_alpha = zero_one_loss(render_pkg["alpha"])
    
    # 4. Material regularization
    l_reg = 0
    # Roughness smoothness
    l_reg += opt.lambda_roughness_smooth * tv_loss(render_pkg['roughness'])
    # Metallic sparsity (most objects non-metallic)
    l_reg += opt.lambda_metallic_sparse * render_pkg['metallic'].mean()
    
    # 5. Combine
    total_loss = (
        l_rgb +
        opt.lambda_normal * l_normal +
        opt.lambda_alpha * l_alpha +
        opt.lambda_material * l_reg
    )
    
    return total_loss, {
        'l_rgb': ll1,
        'l_normal': l_normal,
        'l_alpha': l_alpha,
        'l_reg': l_reg,
    }
```

**Key insights**:
- Use depth-based pseudo-GT as PRIMARY normal supervision (GaussianShader's approach)
- Optional: validate with external model (Marigold) but don't depend on it
- Alpha regularization is CRITICAL for clean geometry
- Don't use delta_normal_reg if you have single normal (GaussianShader uses it because of delta system)

---

### 5. TRAINING DYNAMICS: Learning Rate Strategy

#### 5.1 GaussianShader's Approach

```python
# Environment map (global, affects all pixels)
brdf_mlp_lr = 1.6e-2  # Very high!
brdf_mlp_lr_final = 1.6e-3

# Per-point materials (local, need care)
normal_lr = 2e-4
specular_lr = 2e-4
roughness_lr = 2e-4
```

**Ratio**: Env map LR is ~80x higher than material LRs!

**Scheduling**:
```python
# Exponential decay
lr(t) = lr_init * (lr_final / lr_init)^(t / max_steps)
```

#### 5.2 Your Plan (Not Fully Specified)

```python
# Should be:
# - position_lr: existing, for geometry
# - albedo_lr: new, for diffuse color (similar to position_lr)
# - roughness_lr: new, lower than position_lr
# - metallic_lr: new, lower than position_lr
# - env_mlp_lr: HIGHEST - global lighting
```

#### 5.3 Recommended Configuration

```python
class OptimizationParams:
    # Geometry (unchanged)
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    
    # Appearance (new)
    albedo_lr = 0.001        # Moderate - tied to diffuse appearance
    
    # Materials (new, lower than albedo)
    roughness_lr = 0.0001
    metallic_lr = 0.0001
    normal_lr = 0.0001
    
    # Environment (new, HIGHEST)
    env_mlp_lr_init = 0.01   # Start high
    env_mlp_lr_final = 0.001
    
    # Losses
    lambda_dssim = 0.2
    lambda_normal = 0.2      # 20% weight on normal supervision!
    lambda_alpha = 1e-3      # Small weight on opacity regularization
    lambda_roughness_smooth = 0.01
    lambda_metallic_sparse = 0.01
    
    # Schedules
    normal_reg_from_iter = 0  # Start immediately
```

---

### 6. MATERIAL PARAMETER DECISIONS

#### Decision Tree: What Should Each Parameter Do?

| Parameter | GaussianShader | Your Plan | Choose |
|-----------|---|---|---|
| **Diffuse Color** | In _features_dc (view-dependent) | Separate albedo | **Separate** (cleaner semantics) |
| **Specular** | Direct F0 (3 channels) | F0 = lerp(0.04, albedo, metallic) | **Metallic decomposition** (more interpretable) |
| **Roughness** | Per-pixel (1 channel) | Per-pixel (1 channel) | **Same** |
| **Normals** | Dual with delta | Single | **Single** (sufficient for 2D) |
| **High-frequency details** | SH residuals (features_rest) | Optional fallback | **Keep optional** (capture non-PBR effects) |

#### Activation Functions: Recommended

```python
class GaussianModel:
    def __init__(self, ...):
        # Albedo: ensure [0, 1]
        self.albedo_activation = torch.sigmoid
        
        # Metallic: ensure [0, 1]
        self.metallic_activation = torch.sigmoid
        
        # Roughness: ensure [0, 1] with min to avoid over-shiny
        self.roughness_activation = lambda x: torch.sigmoid(x).clamp(min=0.1)
        # Or: torch.sigmoid with bias
        self.roughness_bias = 0.1  # Offset after sigmoid
        
    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)
    
    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)
    
    @property
    def get_roughness(self):
        # Avoid too-smooth (0.1 ~ 20 degrees)
        return self.roughness_activation(self._roughness) + self.roughness_bias
```

---

### 7. NORMAL SUPERVISION: Depth-Based vs External

#### 7.1 GaussianShader: Depth-Based (Recommended)

```python
# In gaussian_renderer/__init__.py
def render_normal(viewpoint_cam, depth, bg_color, alpha):
    """Compute surface normal from depth"""
    # Use camera intrinsics to unproject depth
    K = viewpoint_cam.intrinsic  # [3, 3]
    
    # Convert depth map to surface normals
    # Typically using Sobel filters: normal = cross(dP/dx, dP/dy)
    normal_ref = normal_from_depth_image(depth, K)
    
    # Alpha composite
    normal_ref = normal_ref * alpha[None, None, ...] + \
                 bg_color[None, None, ...] * (1 - alpha[None, None, ...])
    
    return normal_ref

# Loss
l_normal = predicted_normal_loss(
    render_pkg["normal"],           # Rendered
    render_pkg["normal_ref"],       # From depth
    render_pkg["alpha"]             # Weighted by alpha
)
```

**Advantages**:
- Self-supervised (no external model needed)
- Naturally aligns with rendering geometry
- Differentiable (can supervise through depth)

**Implementation**: Already done in GaussianShader, just need to port to 2DGS

#### 7.2 Alternative: External Model (Marigold)

```python
# Optional: use external normal predictor
if iteration % 100 == 0 and use_marigold:
    with torch.no_grad():
        # Get current rendered image
        current_img = render_pkg['image']
        
        # Run external estimator
        marigold_normal = marigold_model(current_img)
        
        # Transform to world coords
        marigold_normal = transform_to_world(marigold_normal, camera)
        
    # Optional: supervise against it
    l_external = predicted_normal_loss(
        render_pkg["normal"],
        marigold_normal,
        render_pkg["alpha"]
    )
```

**Advantages**: More detailed surface information
**Disadvantages**: Requires external model, inference cost, potential bias

**Recommendation**: Use depth-based as primary, Marigold as validation/optional

---

## 8. FINAL IMPLEMENTATION ROADMAP

### Phase 1: Model Architecture (Week 1)

**File**: `external/2DGS/scene/gaussian_model.py`

```python
class GaussianModel:
    def __init__(self, sh_degree, use_pbr=False):
        # ... existing code ...
        
        self.use_pbr = use_pbr
        if use_pbr:
            # New parameters
            self._albedo = torch.empty(0)       # [N, 3]
            self._roughness = torch.empty(0)    # [N, 1]
            self._metallic = torch.empty(0)     # [N, 1]
            self._normal = torch.empty(0)       # [N, 3]
            
            # Activation functions
            self.albedo_activation = torch.sigmoid
            self.metallic_activation = torch.sigmoid
            self.roughness_activation = torch.sigmoid
            self.roughness_bias = 0.1
    
    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)
    
    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)
    
    @property
    def get_roughness(self):
        return torch.clamp(
            self.roughness_activation(self._roughness) + self.roughness_bias,
            min=0.1, max=0.999
        )
    
    def create_from_pcd(self, pcd, spatial_lr_scale):
        # ... existing code ...
        
        if self.use_pbr:
            # Initialize from point cloud colors
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            self._albedo = nn.Parameter(torch.logit(colors.clamp(0.01, 0.99)).requires_grad_(True))
            
            # Roughness: moderate (0.5 in output space)
            self._roughness = nn.Parameter(torch.zeros((num_pts, 1), device="cuda").requires_grad_(True))
            
            # Metallic: non-metallic (-2.0 -> ~0.1 after sigmoid)
            self._metallic = nn.Parameter(torch.full((num_pts, 1), -2.0, device="cuda").requires_grad_(True))
            
            # Normal: zero (will be learned)
            self._normal = nn.Parameter(torch.zeros((num_pts, 3), device="cuda").requires_grad_(True))
    
    def training_setup(self, training_args):
        # ... existing code ...
        
        if self.use_pbr:
            l.extend([
                {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': list(self.env_mlp.parameters()), 'lr': training_args.env_mlp_lr, "name": "env_mlp"},
            ])
```

### Phase 2: Rendering Pipeline (Week 2)

**File**: `external/2DGS/gaussian_renderer/__init__.py`

```python
def render(viewpoint_camera, pc, pipe, bg_color, ...):
    # ... existing setup ...
    
    if pipe.use_pbr:
        # Compute PBR shading per-gaussian
        gb_pos = pc.get_xyz
        normal = pc.get_normal()
        albedo = pc.get_albedo()
        metallic = pc.get_metallic()
        roughness = pc.get_roughness()
        
        # Compute F0 from metallic
        F0 = torch.lerp(
            torch.full_like(albedo, 0.04),
            albedo,
            metallic
        )
        
        # Shade with environment
        colors_precomp, brdf_extras = pc.env_mlp.shade(
            gb_pos[None, None, ...],
            normal[None, None, ...],
            albedo[None, None, ...],
            F0[None, None, ...],
            roughness[None, None, ...],
            viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1)[None, None, ...]
        )
        colors_precomp = colors_precomp.squeeze()
    else:
        # Original SH-based color
        colors_precomp = ...
    
    # ... render with rasterizer ...
    
    # Compute pseudo-GT normal from depth
    if pipe.use_pbr:
        out_extras["normal_ref"] = render_normal(
            viewpoint_camera,
            out_extras['depth'][0],
            bg_color,
            out_extras['alpha'][0]
        )
```

### Phase 3: Loss Functions (Week 2-3)

**File**: `external/2DGS/utils/loss_utils.py`

Add new losses (same as GaussianShader):
```python
def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Normal supervision from pseudo-GT"""
    # ... (copy from GaussianShader)

def zero_one_loss(img):
    """Encourage opacity to be binary"""
    # ... (copy from GaussianShader)
```

### Phase 4: Training Loop (Week 3-4)

**File**: `external/2DGS/train.py`

```python
def training(dataset, opt, pipe, ...):
    # ... existing setup ...
    
    for iteration in range(1, opt.iterations + 1):
        # ... existing code ...
        
        if pipe.use_pbr:
            # Control normal gradient
            pc.set_requires_grad("normal", state=iteration >= opt.normal_reg_from_iter)
            
            # Build env map mips
            pc.env_mlp.build_mips()
        
        # ... render ...
        
        # Compute losses
        loss = ... # RGB loss (existing)
        
        if pipe.use_pbr:
            if iteration > opt.normal_reg_from_iter:
                loss += opt.lambda_normal * predicted_normal_loss(
                    render_pkg["normal"],
                    render_pkg["normal_ref"],
                    render_pkg["alpha"]
                )
            
            loss += opt.lambda_alpha * zero_one_loss(render_pkg["alpha"])
```

---

## 9. Quick Checklist for Integration

- [ ] Verify 2DGS can load GaussianShader's BRDF LUT (or implement analytical version)
- [ ] Check if nvdiffrast is available in 2DGS dependencies
- [ ] Decide: Use EnvironmentLight class directly or implement minimal version?
- [ ] Decide: Single normal or dual normal? (recommend single for 2D)
- [ ] Decide: Separate albedo parameter or keep in _features_dc? (recommend separate)
- [ ] Implement normal_from_depth_image() if not already available
- [ ] Add new command-line arguments for PBR mode
- [ ] Test initialization with toy data before full training

---

## 10. Potential Issues & Solutions

| Issue | GaussianShader | 2DGS | Solution |
|-------|---|---|---|
| Diffuse always goes to _features_dc | Yes | N/A | Create separate _albedo parameter |
| Dual normal complexity | Yes | N/A | Use single normal for 2D |
| High LR for env map | Yes | TBD | Start with 0.01 for env, 0.0001 for materials |
| BRDF LUT dependency | Yes | TBD | Use same file or implement analytical |
| Normal initialization | Zeros | TBD | Start with zeros, no bias |
| Coordinate system misalignment | Can happen | Careful | Test with known scene first |
| Boundary artifacts in loss | Handled (alpha erosion) | TBD | Implement alpha erosion in loss functions |
| Convergence too slow | Can happen | TBD | Adjust learning rates + normal scheduling |

