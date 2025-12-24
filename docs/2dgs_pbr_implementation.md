# 2DGS PBR 改造实现计划

## 概述

将 2D Gaussian Splatting (2DGS) 的球谐函数 (Spherical Harmonics) 颜色表示替换为基于物理的渲染 (PBR) 材质系统，实现材质-光照分离和重光照能力。

### 当前架构 vs 目标架构

| 方面 | 当前 (SH) | 目标 (PBR) |
|------|-----------|------------|
| 颜色表示 | SH系数 (48 floats/点) | Albedo + Roughness + Metallic + Normal (8 floats/点) |
| 光照 | 隐式编码在SH中 | 显式可学习环境图 (Cubemap) |
| 渲染方式 | Forward (直接输出颜色) | **Deferred** (G-Buffer + 屏幕空间着色) |
| 法线监督 | 无 | **深度自监督** (从渲染深度计算伪GT) |
| 可编辑性 | 无法单独调整材质/光照 | 可独立编辑材质和光照 |

> **关键洞察 (来自 GaussianShader 分析)**:
> - 使用 **深度自监督** 计算伪 GT 法线，无需外部模型 (如 Marigold)
> - **Alpha 正则化** (zero_one_loss) 是关键，鼓励透明度趋向 0 或 1
> - **环境图学习率** 应比材质参数高 ~100 倍
>
> **注**: GaussianShader 使用 Pre-shading，我们选择 Deferred (G-Buffer) 便于重光照和材质编辑

---

## 第一阶段：修改 GaussianModel

### 目标
在每个 Gaussian 点上添加 PBR 材质属性。

### 修改文件
`external/2DGS/scene/gaussian_model.py`

### 具体改动

#### 1.1 添加材质参数属性

```python
# 在 __init__ 中添加
self.use_pbr = use_pbr  # 新增参数控制 PBR 模式

if self.use_pbr:
    self._albedo = torch.empty(0)      # [N, 3] RGB基础颜色
    self._roughness = torch.empty(0)   # [N, 1] 粗糙度 0-1
    self._metallic = torch.empty(0)    # [N, 1] 金属度 0-1
    self._normal = torch.empty(0)      # [N, 3] 表面法线偏移量

    # 激活函数
    self.albedo_activation = torch.sigmoid
    self.metallic_activation = torch.sigmoid
    self.roughness_activation = torch.sigmoid
    self.roughness_min = 0.1  # 避免过于光滑 (GaussianShader 经验)
```

#### 1.2 添加属性访问器

```python
@property
def get_albedo(self):
    return self.albedo_activation(self._albedo)  # 约束到 [0, 1]

@property
def get_roughness(self):
    # 最小粗糙度 0.1，避免过于光滑导致的伪影
    return torch.clamp(
        self.roughness_activation(self._roughness) + self.roughness_min,
        min=0.1, max=0.999
    )

@property
def get_metallic(self):
    return self.metallic_activation(self._metallic)  # 约束到 [0, 1]

@property
def get_normal(self):
    """获取表面法线"""
    # 方案1: 从2DGS的协方差矩阵计算 (min axis)
    normal_axis = self.get_minimum_axis  # 2DGS 已有

    # 方案2: 添加可学习的偏移量
    if hasattr(self, '_normal') and self._normal.shape[0] > 0:
        normal = normal_axis + self._normal
        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-7)
        return normal
    return normal_axis
```

#### 1.3 修改 create_from_pcd

```python
def create_from_pcd(self, pcd, spatial_lr_scale):
    # ... 现有代码 ...
    num_pts = len(pcd.points)

    if self.use_pbr:
        # 从点云颜色初始化 albedo
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # 反sigmoid，使得sigmoid后恢复原值
        self._albedo = nn.Parameter(
            torch.logit(colors.clamp(0.01, 0.99)).requires_grad_(True)
        )

        # roughness 初始化为中等偏粗 (GaussianShader 使用 0.6)
        # sigmoid(0.4) ≈ 0.6
        self._roughness = nn.Parameter(
            torch.full((num_pts, 1), 0.4, device="cuda").requires_grad_(True)
        )

        # metallic 初始化为非金属 (sigmoid(-2) ≈ 0.12)
        self._metallic = nn.Parameter(
            torch.full((num_pts, 1), -2.0, device="cuda").requires_grad_(True)
        )

        # normal 初始化为零 (纯粹从监督学习)
        self._normal = nn.Parameter(
            torch.zeros((num_pts, 3), device="cuda").requires_grad_(True)
        )
```

#### 1.4 修改优化器设置 (training_setup)

**关键洞察**: 环境图需要 ~100x 更高的学习率

```python
def training_setup(self, training_args):
    # ... 现有参数 ...

    if self.use_pbr:
        l.extend([
            # 材质参数 - 较低学习率
            {'params': [self._albedo], 'lr': 0.001, "name": "albedo"},
            {'params': [self._roughness], 'lr': 0.0002, "name": "roughness"},
            {'params': [self._metallic], 'lr': 0.0002, "name": "metallic"},
            {'params': [self._normal], 'lr': 0.0002, "name": "normal"},

            # 环境图 - 高学习率 (全局参数，影响所有像素)
            {'params': list(self.env_light.parameters()),
             'lr': 0.016, "name": "env_light"},  # ~100x 材质
        ])
```

#### 1.5 修改 save_ply / load_ply

添加 albedo, roughness, metallic, normal 的保存和加载。

#### 1.6 关于 SH 的处理

**推荐**: SH 作为高频残差项（GaussianShader 做法）

```python
# 保留 _features_rest 用于捕捉非 PBR 效果
# 例如：各向异性反射、次表面散射等
if self.use_pbr:
    # PBR 颜色 + 可选的 SH 残差
    final_color = pbr_color
    if self.use_sh_residual:
        sh_residual = eval_sh(self.active_sh_degree, self._features_rest, dir_pp)
        final_color = final_color + 0.1 * sh_residual
```

---

## 第二阶段：渲染管线改造 (Deferred Shading)

### 目标
实现 **Deferred Shading** 架构：先光栅化 G-Buffer，再在屏幕空间进行 PBR 着色。

> **为什么是 Deferred?**
> - G-Buffer 可复用于重光照、材质编辑
> - 调试更直观（可单独查看各属性图）
> - 符合标准延迟渲染流程

### 修改文件
`external/2DGS/gaussian_renderer/__init__.py`

### 具体改动

#### 2.1 Deferred 渲染流程

```python
def render(viewpoint_camera, pc, pipe, bg_color, ...):
    """PBR 渲染主函数 (Deferred Shading)"""

    # ===== 1. 光栅化 G-Buffer =====
    # 先渲染各材质属性图到屏幕空间
    gbuffer = render_gbuffer(viewpoint_camera, pc, pipe, bg_color)
    # gbuffer 包含: albedo, normal, roughness, metallic, depth, alpha

    if pipe.use_pbr:
        # ===== 2. 屏幕空间 PBR 着色 =====
        camera_pos = viewpoint_camera.camera_center

        # 从深度图重建世界坐标
        world_pos = depth_to_world_pos(
            gbuffer['depth'],
            viewpoint_camera
        )

        # 计算视角方向 (屏幕空间每个像素)
        view_dir = F.normalize(camera_pos - world_pos, dim=-1)

        # PBR 着色
        rendered_image = pbr_shading(
            albedo=gbuffer['albedo'],
            normal=gbuffer['normal'],
            roughness=gbuffer['roughness'],
            metallic=gbuffer['metallic'],
            view_dir=view_dir,
            env_light=pc.env_light
        )

        # ===== 3. 计算深度伪 GT 法线 (自监督) =====
        gbuffer["normal_ref"] = compute_normal_from_depth(
            gbuffer['depth'],
            viewpoint_camera
        )

    else:
        # 原始 SH 渲染
        rendered_image = render_sh(viewpoint_camera, pc, pipe, bg_color)

    return {"image": rendered_image, **gbuffer}
```

#### 2.2 渲染 G-Buffer

```python
def render_gbuffer(viewpoint_camera, pc, pipe, bg_color):
    """光栅化各材质属性到屏幕空间"""
    extras = {}

    # 法线图
    extras['normal'] = rasterizer(
        means3D=means3D,
        colors_precomp=normal * 0.5 + 0.5,  # [-1,1] -> [0,1]
        ...
    )[0]

    # Albedo 图
    extras['albedo'] = rasterizer(
        means3D=means3D,
        colors_precomp=albedo,
        ...
    )[0]

    # Roughness (扩展为3通道)
    extras['roughness'] = rasterizer(
        means3D=means3D,
        colors_precomp=roughness.repeat(1, 3),
        ...
    )[0][:1]

    # Metallic
    extras['metallic'] = rasterizer(
        means3D=means3D,
        colors_precomp=metallic.repeat(1, 3),
        ...
    )[0][:1]

    return extras
```

#### 2.3 深度到法线转换 (自监督关键!)

```python
def compute_normal_from_depth(depth, camera):
    """从渲染深度计算表面法线 (GaussianShader 的自监督方法)"""
    # 使用相机内参将深度图转换为点云
    K = camera.K  # [3, 3] 内参矩阵

    # 计算深度梯度 (Sobel 滤波器)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=depth.dtype, device=depth.device) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=depth.dtype, device=depth.device) / 8.0

    grad_x = F.conv2d(depth[None, None], sobel_x[None, None], padding=1)
    grad_y = F.conv2d(depth[None, None], sobel_y[None, None], padding=1)

    # 从梯度计算法线
    normal = torch.stack([
        -grad_x.squeeze(),
        -grad_y.squeeze(),
        torch.ones_like(depth)
    ], dim=-1)

    normal = F.normalize(normal, dim=-1)

    # 转换到世界坐标系
    R = camera.R.T
    normal = normal @ R

    return normal
```

---

## 第三阶段：PBR Shading 实现

### 目标
实现 Split-Sum IBL 着色。

### 新建文件
`external/2DGS/utils/pbr_utils.py`

### 3.1 可学习环境光 (Cubemap)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvironmentLight(nn.Module):
    """可学习的 HDR 环境光照 (来自 GaussianShader)"""

    def __init__(self, base_res=64):
        super().__init__()
        # 立方体贴图: [6 faces, H, W, 3 channels]
        self.base = nn.Parameter(
            torch.rand(6, base_res, base_res, 3) * 0.5 + 0.25  # 初始化为中灰
        )

        # 预计算 BRDF LUT (256x256x2)
        # 可从 GaussianShader 加载，或使用解析近似
        self._FG_LUT = self._load_or_compute_brdf_lut()

        self.specular_mips = None
        self.diffuse = None

    def build_mips(self):
        """构建 Mipmap 金字塔 (用于粗糙度采样)"""
        self.specular_mips = [self.base]
        current = self.base

        while current.shape[1] > 16:
            # 下采样
            current = F.avg_pool2d(
                current.permute(0, 3, 1, 2),
                kernel_size=2
            ).permute(0, 2, 3, 1)
            self.specular_mips.append(current)

        # 最低级 = 漫反射辐照度
        self.diffuse = self._compute_diffuse_irradiance(self.specular_mips[-1])

    def get_mip_level(self, roughness):
        """粗糙度映射到 Mip 级别"""
        max_mip = len(self.specular_mips) - 1
        return roughness * max_mip

    def shade(self, positions, normals, albedo, F0, roughness, view_dir):
        """Split-Sum IBL 着色"""
        # 确保 mip 已构建
        if self.specular_mips is None:
            self.build_mips()

        # 1. 漫反射: 从法线方向采样漫反射辐照度
        diffuse_irradiance = self._sample_cubemap(self.diffuse, normals)
        kd = (1.0 - F0) * (1.0 - roughness)  # 近似 kD
        diffuse = kd * albedo * diffuse_irradiance

        # 2. 镜面反射
        # 计算反射方向
        reflect_dir = 2.0 * torch.sum(view_dir * normals, dim=-1, keepdim=True) * normals - view_dir

        # 根据粗糙度选择 mip 级别
        mip = self.get_mip_level(roughness)
        specular_env = self._sample_cubemap_mip(reflect_dir, mip)

        # BRDF LUT 查询
        NdotV = torch.clamp(torch.sum(normals * view_dir, dim=-1, keepdim=True), 0, 1)
        fg = self._sample_brdf_lut(NdotV, roughness)

        # Split-Sum: L_spec = Li * (F0 * fg.x + fg.y)
        specular = specular_env * (F0 * fg[..., :1] + fg[..., 1:2])

        # 3. 最终颜色
        color = diffuse + specular

        return color, {'diffuse': diffuse, 'specular': specular}
```

### 3.2 解析 BRDF (无 LUT 备选)

```python
def analytical_pbr_shading(albedo, normal, roughness, metallic, view_dir,
                           light_dir, light_color):
    """
    解析 PBR 着色 (单光源)
    用于快速调试，无需 BRDF LUT
    """
    # 半角向量
    H = F.normalize(view_dir + light_dir, dim=-1)

    NdotL = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), 0, 1)
    NdotV = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), 0, 1)
    NdotH = torch.clamp(torch.sum(normal * H, dim=-1, keepdim=True), 0, 1)
    VdotH = torch.clamp(torch.sum(view_dir * H, dim=-1, keepdim=True), 0, 1)

    # F0 (基础反射率)
    F0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)

    # Fresnel - Schlick 近似
    F = F0 + (1.0 - F0) * torch.pow(1.0 - VdotH, 5.0)

    # Distribution - GGX
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
    D = alpha2 / (math.pi * denom * denom + 1e-7)

    # Geometry - Smith GGX
    k = (roughness + 1.0) ** 2 / 8.0
    G_V = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
    G_L = NdotL / (NdotL * (1.0 - k) + k + 1e-7)
    G = G_V * G_L

    # Specular BRDF
    specular = D * F * G / (4.0 * NdotV * NdotL + 1e-7)

    # Diffuse - Lambert
    kD = (1.0 - F) * (1.0 - metallic)
    diffuse = kD * albedo / math.pi

    # 最终颜色
    color = (diffuse + specular) * light_color * NdotL

    return color
```

---

## 第四阶段：Loss 函数设计

### 修改文件
`external/2DGS/train.py` 和 `external/2DGS/utils/loss_utils.py`

### 4.1 新增 Loss 函数

```python
# loss_utils.py

def zero_one_loss(alpha):
    """
    Alpha 正则化 (来自 GaussianShader)
    鼓励透明度趋向 0 或 1 (二值化)
    """
    eps = 1e-3
    val = torch.clamp(alpha, eps, 1 - eps)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_normal_loss(normal_pred, normal_ref, alpha=None):
    """
    法线监督损失 (来自 ref-NeRF)
    使用深度计算的伪 GT 作为监督
    """
    # 可选: 侵蚀 alpha 边界，避免边缘伪影
    if alpha is not None:
        weight = erode_alpha(alpha, size=4)
    else:
        weight = torch.ones_like(normal_pred[..., 0])

    # 余弦相似度损失
    cos_sim = torch.sum(normal_pred * normal_ref.detach(), dim=-1)
    loss = (weight * (1.0 - cos_sim)).mean()

    return loss

def erode_alpha(alpha, size=4):
    """侵蚀 alpha 边界"""
    kernel = torch.ones(1, 1, size, size, device=alpha.device) / (size * size)
    eroded = F.conv2d(alpha[None, None], kernel, padding=size//2)
    return (eroded > 0.99).float().squeeze()

def tv_loss(img):
    """Total Variation 平滑损失"""
    diff_x = img[:, :, 1:] - img[:, :, :-1]
    diff_y = img[:, 1:, :] - img[:, :-1, :]
    return torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
```

### 4.2 完整 Loss 组合

```python
# train.py

def compute_pbr_loss(render_pkg, gt_image, opt, iteration):
    """
    PBR 训练损失

    总 Loss = L_rgb + λ_normal * L_normal + λ_alpha * L_alpha + L_reg
    """
    losses = {}

    # ===== 1. RGB 重建损失 (标准) =====
    L_l1 = l1_loss(render_pkg['image'], gt_image)
    L_ssim = 1.0 - ssim(render_pkg['image'], gt_image)
    losses['rgb'] = (1.0 - opt.lambda_dssim) * L_l1 + opt.lambda_dssim * L_ssim

    # ===== 2. 法线监督损失 (自监督) =====
    # 使用深度计算的伪 GT，无需外部模型
    if iteration > opt.normal_reg_from_iter:
        losses['normal'] = predicted_normal_loss(
            render_pkg['normal'],
            render_pkg['normal_ref'],  # 从深度计算
            render_pkg['alpha']
        )
    else:
        losses['normal'] = 0

    # ===== 3. Alpha 正则化 (关键!) =====
    losses['alpha'] = zero_one_loss(render_pkg['alpha'])

    # ===== 4. 材质正则化 =====
    losses['reg'] = 0
    # 粗糙度平滑
    losses['reg'] += opt.lambda_roughness_smooth * tv_loss(render_pkg['roughness'])
    # 金属度稀疏 (大部分物体非金属)
    losses['reg'] += opt.lambda_metallic_sparse * render_pkg['metallic'].mean()

    # ===== 5. 总损失 =====
    total_loss = (
        losses['rgb'] +
        opt.lambda_normal * losses['normal'] +
        opt.lambda_alpha * losses['alpha'] +
        losses['reg']
    )

    return total_loss, losses
```

### 4.3 推荐超参数

```python
# arguments/__init__.py

class OptimizationParams:
    # === 现有参数 ===
    position_lr_init = 0.00016
    position_lr_final = 0.0000016

    # === PBR 材质学习率 (较低) ===
    albedo_lr = 0.001
    roughness_lr = 0.0002
    metallic_lr = 0.0002
    normal_lr = 0.0002

    # === 环境图学习率 (高!) ===
    env_light_lr_init = 0.016     # ~100x 材质
    env_light_lr_final = 0.0016

    # === Loss 权重 ===
    lambda_dssim = 0.2
    lambda_normal = 0.2          # 20% (重要!)
    lambda_alpha = 0.001         # 小权重
    lambda_roughness_smooth = 0.01
    lambda_metallic_sparse = 0.01

    # === 训练调度 ===
    normal_reg_from_iter = 0     # 从第0轮开始法线监督
```

---

## 关键代码位置参考

| 功能 | 文件 | 关键函数/行号 |
|------|------|--------------|
| Gaussian 模型定义 | `scene/gaussian_model.py` | `GaussianModel` 类 |
| SH 系数存储 | `scene/gaussian_model.py` | `_features_dc`, `_features_rest` (~L141) |
| SH 评估 | `utils/sh_utils.py` | `eval_sh()` (~L57) |
| 渲染入口 | `gaussian_renderer/__init__.py` | `render()` (~L19) |
| CUDA SH计算 | `cuda_rasterizer/forward.cu` | `computeColorFromSH()` (~L18) |
| 训练循环 | `train.py` | `training()` |
| 优化器设置 | `scene/gaussian_model.py` | `training_setup()` |

### GaussianShader 参考

| 功能 | 文件 | 说明 |
|------|------|------|
| PBR 参数定义 | `scene/gaussian_model.py:52-58` | _normal, _specular, _roughness |
| 环境图着色 | `scene/NVDIFFREC/light.py:128-174` | Split-Sum IBL |
| 法线监督 | `train.py:86-88` | predicted_normal_loss |
| Alpha 正则化 | `utils/loss_utils.py:71-75` | zero_one_loss |
| BRDF LUT | `scene/NVDIFFREC/light.py` | 预计算 256x256x2 |

---

## 实现顺序建议

```
Week 1: 第一阶段 - 模型结构
├── 添加 PBR 参数到 GaussianModel (_albedo, _roughness, _metallic, _normal)
├── 实现属性访问器 (get_albedo, get_roughness, 等)
├── 修改 save_ply / load_ply
├── 修改 training_setup (优化器配置)
└── 验证参数能正常保存加载

Week 2: 第二阶段 - 渲染管线
├── 实现 render_gbuffer() 光栅化各材质属性
├── 实现 pbr_shading() 屏幕空间着色
├── 实现 EnvironmentLight 类 (可从 GaussianShader 移植)
├── 实现 compute_normal_from_depth() (深度自监督)
└── 验证各属性图和着色结果正确

Week 3: 第三阶段 - Loss 函数
├── 实现 zero_one_loss (Alpha 正则化)
├── 实现 predicted_normal_loss (法线监督)
├── 添加材质正则化 (roughness smooth, metallic sparse)
├── 修改 train.py 集成新 Loss
└── 调整超参数

Week 4: 第四阶段 - 完整训练
├── 端到端训练测试
├── 学习率调优 (特别是 env_light 的高 LR)
├── 可视化调试 (G-Buffer, 法线, 材质图)
├── 对比实验 (SH vs PBR)
└── 效果调优
```

---

## 关键洞察 (来自 GaussianShader)

### 1. Pre-shading vs Deferred (G-Buffer)
- **GaussianShader 使用 Pre-shading**: 先着色后光栅化
  - `shade(G1)*α1 + shade(G2)*α2` (每个 Gaussian 独立着色)
- **我们选择 Deferred**: 先光栅化 G-Buffer，再屏幕空间着色
  - `shade(G1*α1 + G2*α2)` (混合后再着色)
  - 优点: G-Buffer 可用于重光照、材质编辑、调试
  - 代价: 混合后着色可能丢失部分高频细节

### 2. 法线监督策略
- **使用深度自监督**: 从渲染深度计算伪 GT 法线
- **无需外部模型** (如 Marigold)，完全自监督
- 关键: 侵蚀 alpha 边界 (size=4) 避免边缘伪影

### 3. 学习率层次
- **环境图**: 0.016 (高! 全局参数)
- **材质**: 0.0002 (低，局部参数)
- **比例**: ~100:1

### 4. Alpha 正则化
- **zero_one_loss** 鼓励透明度趋向 0 或 1
- 对几何质量**至关重要**

### 5. 粗糙度初始化
- 使用 **0.6** (中等偏粗) 而非 0.5
- 避免初始过于光滑导致的伪影

### 6. 单法线 vs 双法线
- **2DGS 推荐单法线**: 对象通常单面
- GaussianShader 的双法线系统用于处理背面，2DGS 可简化

---

## 注意事项

1. **渐进式开发**: 先用解析 BRDF 验证流程，再使用 Split-Sum + LUT
2. **保留 SH 作为 fallback**: 初期调试时可保留 SH，便于对比
3. **坐标系统一**: 法线、光照方向务必在同一坐标系
4. **数值稳定性**: BRDF 分母加 epsilon 防止除零
5. **学习率调优**: 环境图需要高学习率!
6. **可视化调试**: 每个阶段都输出中间结果可视化检查

---

## 依赖项

- **nvdiffrast** (可选): 用于高效立方体贴图采样
- **BRDF LUT**: 可从 GaussianShader 加载 `bsdf_256_256.bin`，或使用解析近似

---

## 参考资源

- [GaussianShader](https://github.com/Asparagus15/GaussianShader) - 3DGS PBR 实现 (主要参考)
- [GS-IR](https://github.com/lzhnb/GS-IR) - 逆向渲染
- [Relightable 3D Gaussian](https://github.com/NJU-3DV/Relightable3DGaussian) - 重光照
- [Learn OpenGL PBR](https://learnopengl.com/PBR/Theory) - PBR 理论
