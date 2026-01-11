# 2DGS-PBR Mono Normal Supervision 完整分析报告

## 执行时间
2025-12-31

---

## 1. GT 法线生成流程分析 (render_gt_open3d.py)

### 1.1 Mesh 法线坐标系定义

**位置**：render_gt_open3d.py, 第 163-168 行

```python
mesh_normal = o3d.geometry.TriangleMesh(mesh)
normals = np.asarray(mesh_normal.vertex_normals)  # World space normals

# Map [-1, 1] -> [0, 1]
colors = (normals + 1.0) * 0.5
mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)
```

**关键发现**：
- `mesh.vertex_normals` 返回**世界空间法线** (World Space)
- Open3D 的约定：Y 向下，Z 向前 (与 OpenGL 不同)
- 法线直接来自几何计算，无特殊坐标系转换

### 1.2 法线编码为颜色

**编码方法**：
```
normal_encoded = (normal_world + 1.0) * 0.5
范围：[-1, 1] → [0, 1]
存储：8-bit PNG，uint8 (0-255)
```

**编码公式**：
- X component: `(nx + 1.0) * 0.5 * 255` → [0, 255]
- Y component: `(ny + 1.0) * 0.5 * 255` → [0, 255]  
- Z component: `(nz + 1.0) * 0.5 * 255` → [0, 255]

### 1.3 背景值

**位置**：render_gt_open3d.py, 第 202 行

```python
renderer.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
```

**背景值**：
- RGB: (0.0, 0.0, 0.0)
- 编码后：`(0 + 1.0) * 0.5 = 0.5` → PNG 值 127.5 ≈ **128**
- 含义：代表世界空间法线 (-1, -1, -1)（不可能的法线值）

### 1.4 坐标系变换：flip_yz 的作用

**位置**：render_gt_open3d.py, 第 211-224 行

```python
# Coordinate system conversion: OpenGL/Blender -> Open3D
# OpenGL: Y up, -Z forward (camera looks down -Z)
# Open3D: Y down, +Z forward (camera looks down +Z)
# We need to flip Y and Z axes

flip_yz = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

c2w_o3d = c2w @ flip_yz
w2c = np.linalg.inv(c2w_o3d)
```

**变换分析**：
1. `c2w` 来自 transforms_train.json，是 **OpenGL 格式** (Y 向上，Z 向后)
2. `flip_yz` 矩阵翻转 Y 和 Z 轴
3. `c2w_o3d` 是转换后的矩阵，适配 Open3D (Y 向下，Z 向前)
4. `w2c = inv(c2w_o3d)` 用于 Open3D 渲染器

**关键点**：
- 这里只是**相机坐标系转换**
- Mesh 和法线已经在**世界坐标系**中
- flip_yz 不改变 mesh 本身，只改变相机观看方向

---

## 2. GT 法线加载与解码 (scene/__init__.py)

### 2.1 法线加载流程

**位置**：scene/__init__.py, 第 145-188 行 (_load_single_pseudo_gt 函数)

```python
if normal_exists:
    n_path = os.path.join(normal_root, f"{filename}.png")
    if os.path.exists(n_path):
        normal_img = cv2.imread(n_path)  # BGR 格式
        if normal_img is not None:
            # BGR -> RGB
            normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
            # Convert to float tensor [0, 1] -> [-1, 1]
            normal_tensor = torch.from_numpy(normal_img.astype(np.float32) / 255.0)
            normal_tensor = normal_tensor * 2.0 - 1.0  # ✓ 正确解码
            # [H, W, 3] -> [3, H, W]
            normal_tensor = normal_tensor.permute(2, 0, 1)
            # ... resizing ...
            cam.pseudo_gt_normal = normal_tensor  # [3, H, W]
```

### 2.2 解码公式验证

**编码公式**（train_pbr.py, 第 75-79 行）：
```python
normal_tensor = torch.from_numpy(normal_img.astype(np.float32) / 255.0)  # [0, 1]
normal_tensor = normal_tensor * 2.0 - 1.0  # [-1, 1] ✓ 正确
```

**逆过程验证**：
```
原始法线: n ∈ [-1, 1]
编码 1: (n + 1.0) * 0.5 → [0, 1]
编码 2: * 255 → [0, 255] (PNG uint8)
加载 1: / 255.0 → [0, 1]
解码: * 2.0 - 1.0 = 2*[0,1] - 1 = [-1, 1] ✓ 完美复原
```

### 2.3 BGR/RGB 转换

**位置**：scene/__init__.py, 第 174 行

```python
normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
```

**分析**：
- OpenCV 默认读入 BGR 格式
- 转换为 RGB 确保正确的通道顺序
- 由于 (R, G, B) 对应 (X, Y, Z)，顺序必须正确

**关键点**：✓ 正确无误

---

## 3. 2DGS 渲染法线计算 (gaussian_renderer/__init__.py)

### 3.1 rasterizer 输出的坐标系

**背景知识**：
- 2DGS 的光栅化器输出**视图空间法线**（View Space）
- 即相对于相机坐标系的法线

**位置**：gaussian_renderer/__init__.py, 第 183-184 行

```python
# get normal map
# transform normal from view space to world space
render_normal = allmap[2:5]
render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
```

### 3.2 world_view_transform 的计算

**位置**：scene/cameras.py, 第 61 行

```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
```

**getWorld2View2 实现**：scene/dataset_readers.py & utils/graphics_utils.py

```python
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # ← R 已转置存储
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)  # ← 这里的 Rt 是 W2C
    return np.float32(Rt)  # 返回 World2View (W2C) 矩阵
```

### 3.3 变换 `(render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)` 分析

**这里是最关键的部分！**

```python
render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
```

**分解步骤**：

1. `render_normal` 初始形状：[3, H, W]（视图空间法线）

2. `render_normal.permute(1,2,0)`：[3, H, W] → [H, W, 3]
   - 转换为每像素一个法线向量的形式

3. `viewpoint_camera.world_view_transform[:3,:3]`：[3, 3] 矩阵
   - 这是 W2C（World-to-Camera）的旋转部分
   - **关键问题**：为什么要用 `.T`（转置）？

4. **矩阵乘法** `[H, W, 3] @ [3, 3]`：
   ```
   每个像素的法线 n_view [1, 3] @ W2C^T [3, 3] = n_world [1, 3]
   ```

5. `permute(2,0,1)`：[H, W, 3] → [3, H, W]（转回原格式）

### 3.4 坐标变换数学验证

**设定**：
- `n_view`：视图空间法线
- `W2C`：World-to-Camera 变换矩阵
- `n_world`：世界空间法线

**法线变换规则**：
```
法线不是位置向量，不能直接用 W2C 变换！
正确做法：n_world = (W2C^(-T)) @ n_view = (C2W^T) @ n_view

其中：
W2C^(-T) = (W2C^(-1))^T = (C2W)^T
```

**代码分析**：
```python
# 当前代码：
n_world = n_view @ W2C^T

# 数学要求：
n_world = C2W^T @ n_view = W2C^(-T) @ n_view
```

**问题检测**：

W2C 的逆矩阵是：
```python
W2C^(-1) = C2W
W2C^(-T) = (W2C^(-1))^T = C2W^T
```

而代码使用的是：
```python
W2C^T  # ← 这是错的！应该是 W2C^(-T) = C2W^T
```

**这是一个潜在的bug！**

### 3.5 正确的法线变换应该是

```python
# 错误（当前代码）：
render_normal = (render_normal.permute(1,2,0) @ 
                 (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

# 正确做法：
W2C = viewpoint_camera.world_view_transform[:3,:3]
C2W = torch.inverse(W2C)  # 或使用预先计算的 C2W
render_normal = (render_normal.permute(1,2,0) @ 
                 C2W.T).permute(2,0,1)
```

---

## 4. 相机参数来源与坐标系

### 4.1 transforms_train.json 中的 c2w 矩阵

**位置**：render_gt_open3d.py, 第 38-40 行

```python
# Transform matrix is 4x4 C2W (Camera to World)
# Coordinate system: OpenGL/Blender (Right=X, Up=Y, Back=Z) -> Camera looks down -Z
c2w = np.array(frame["transform_matrix"])
```

**坐标系定义**：
- **OpenGL/Blender 坐标系**
- X 向右
- Y 向上
- Z 向后（相机看向 -Z）

### 4.2 R, T 的提取 (scene/dataset_readers.py)

**位置**：scene/dataset_readers.py, 第 194-202 行

```python
# NeRF 'transform_matrix' is a camera-to-world transform
c2w = np.array(frame["transform_matrix"])
# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
c2w[:3, 1:3] *= -1

# get the world-to-camera transform and set R, T
w2c = np.linalg.inv(c2w)
R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
T = w2c[:3, 3]
```

**关键点**：
1. `c2w[:3, 1:3] *= -1` → 翻转 Y 和 Z，将 OpenGL 坐标系转为 COLMAP 坐标系
2. `w2c = inv(c2w)` → 反演得到世界到相机
3. `R = w2c[:3,:3].T` → 转置（CUDA GLM 约定）
4. `T = w2c[:3, 3]` → 平移向量

### 4.3 坐标系一致性问题

**问题识别**：

| 步骤 | 坐标系 | 法线坐标系 |
|------|--------|-----------|
| transforms_train.json | OpenGL (Y up, Z back) | 世界空间 (OpenGL) |
| render_gt_open3d.py | flip_yz 转换为 Open3D | 世界空间 (Open3D) |
| dataset_readers.py | c2w[:3,1:3] *= -1 | 转为 COLMAP |
| 2DGS 渲染 | COLMAP/内部 | 视图空间 |
| GT 法线 | Open3D | 世界空间 (Open3D) |

**关键发现**：
- GT 法线在 **Open3D 世界坐标系** 中
- 2DGS 使用的是 **COLMAP 坐标系** 的变换
- 坐标系不一致！

---

## 5. Loss 计算分析 (train_pbr.py)

### 5.1 Mono Normal Loss 计算

**位置**：train_pbr.py, 第 289-301 行

```python
if gt_normal is not None and weights['mono_normal'] > 0:
    with profiler.profile("mono_normal_transfer"):
        gt_normal = gt_normal.cuda()
    pred_normal = render_pkg["rend_normal"]

    with profiler.profile("mono_normal_loss"):
        pred_norm = F.normalize(pred_normal, dim=0)
        gt_norm = F.normalize(gt_normal, dim=0)
        cosine_sim = (pred_norm * gt_norm).sum(dim=0)
        valid_mask = (gt_normal.abs().sum(dim=0) > 0.1)

        if valid_mask.sum() > 0:
            loss_mono_normal = weights['mono_normal'] * (1.0 - cosine_sim[valid_mask]).mean()
```

### 5.2 valid_mask 逻辑分析

**代码**：
```python
valid_mask = (gt_normal.abs().sum(dim=0) > 0.1)
```

**工作原理**：
```
对每个像素：
  magnitude = |gt_normal_x| + |gt_normal_y| + |gt_normal_z|
  valid = magnitude > 0.1
```

**问题分析**：

1. **背景值检测**：
   - 背景 PNG 值 (128, 128, 128)
   - 解码：`128/255*2 - 1 = 0.003` (每分量)
   - 和：`3 * 0.003 ≈ 0.01 < 0.1` ✓ 正确过滤

2. **阈值 0.1 的合理性**：
   - 对于单位法线向量：`magnitude = sqrt(nx^2 + ny^2 + nz^2) = 1`
   - 但这里用的是 **L1 范数**（绝对值和）
   - 对于 (-1, -1, -1) 归一化后：`(-1/sqrt(3), -1/sqrt(3), -1/sqrt(3))`
   - L1 范数：`3/sqrt(3) ≈ 1.73`
   - 阈值 0.1 过于严格吗？

**更好的检测**：
```python
# 使用 L2 范数会更好
valid_mask = (torch.norm(gt_normal, dim=0) > 0.1)
# 或使用最大分量
valid_mask = (gt_normal.abs().max(dim=0)[0] > 0.1)
```

### 5.3 Cosine Similarity 计算

**代码**：
```python
pred_norm = F.normalize(pred_normal, dim=0)  # 沿第 0 维归一化
gt_norm = F.normalize(gt_normal, dim=0)      # 沿第 0 维归一化
cosine_sim = (pred_norm * gt_norm).sum(dim=0)
```

**验证**：
```
对每个像素点 (i, j)：
  pred_norm[:, i, j] = [nx, ny, nz] / ||(nx, ny, nz)||
  gt_norm[:, i, j] = [nx, ny, nz] / ||(nx, ny, nz)||
  cosine_sim[i, j] = sum(pred_norm[:, i, j] * gt_norm[:, i, j])
                   = cos(angle between vectors) ✓ 正确
```

---

## 关键发现总结

### 🔴 严重问题

#### 问题 1: 法线坐标系不匹配

**症状**：Mono normal supervision 效果差

**根本原因**：
```
GT 法线坐标系：Open3D (Y down, Z forward) in World Space
2DGS 法线坐标系：COLMAP (经过坐标系变换) in World Space
```

- `render_gt_open3d.py` 使用 `flip_yz` 转换相机，但 mesh 法线已经在 Open3D 世界坐标系
- `scene/dataset_readers.py` 在加载相机时做了 `c2w[:3, 1:3] *= -1` 坐标系转换
- 导致两套坐标系的法线无法对齐

#### 问题 2: 法线变换公式错误

**症状**：渲染法线转换到世界空间后方向不对

**根本原因**：
```python
# 错误的做法（当前代码）
n_world = n_view @ W2C.T

# 正确的做法
n_world = n_view @ C2W.T  # 或 n_view @ W2C^(-T)
```

法线是方向向量，需要用逆转置变换！

### 🟡 中等问题

#### 问题 3: valid_mask 检测不够鲁棒

**问题**：使用 L1 范数作为背景检测可能不够准确

**改进方案**：
```python
# 选项 1：使用 L2 范数
valid_mask = (torch.norm(gt_normal, dim=0) > 0.1)

# 选项 2：使用最大分量
valid_mask = (gt_normal.abs().max(dim=0)[0] > 0.1)

# 选项 3：使用 alpha map（如果可用）
# valid_mask = (alpha_map[0] > 0.5)
```

### 🟢 设计良好的方面

1. ✓ 颜色编码/解码公式完全正确
2. ✓ BGR/RGB 转换正确
3. ✓ Cosine similarity 计算正确
4. ✓ 背景值选择合理（PNG 值 128）

---

## 完整修复方案

### 修复 1: 统一坐标系

**方案 A**：统一使用 OpenGL 坐标系（推荐）

在 `render_gt_open3d.py` 中，**不要**使用 flip_yz：

```python
# 删除这些行：
# flip_yz = np.array([...])
# c2w_o3d = c2w @ flip_yz
# w2c = np.linalg.inv(c2w_o3d)

# 直接使用：
w2c = np.linalg.inv(c2w)
renderer.setup_camera(intrinsic, w2c)
```

同时在 `scene/dataset_readers.py` 中不要做坐标系转换：

```python
# 删除这行：
# c2w[:3, 1:3] *= -1

# 直接提取：
w2c = np.linalg.inv(c2w)
```

### 修复 2: 修正法线变换公式

**位置**：gaussian_renderer/__init__.py, 第 183-184 行

```python
# 修改前：
render_normal = (render_normal.permute(1,2,0) @ 
                 (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

# 修改后：
W2C_rot = viewpoint_camera.world_view_transform[:3,:3]
C2W_rot = torch.inverse(W2C_rot)
render_normal = (render_normal.permute(1,2,0) @ 
                 C2W_rot.T).permute(2,0,1)
```

### 修复 3: 改进 valid_mask

**位置**：train_pbr.py, 第 298 行

```python
# 修改前：
valid_mask = (gt_normal.abs().sum(dim=0) > 0.1)

# 修改后：
valid_mask = (torch.norm(gt_normal, dim=0) > 0.1)
```

---

## 验证建议

### 调试代码

```python
# 在 train_pbr.py 中添加调试输出
if iteration == loss_scheduler.stages.stage2_end:  # mono supervision 刚开始
    print("=== Normal Supervision Debug ===")
    print(f"GT normal range: {gt_normal.min():.4f} ~ {gt_normal.max():.4f}")
    print(f"GT normal mean: {gt_normal.mean(dim=(1,2))}")  # 应该接近 (0, 0, 0)
    print(f"Pred normal range: {pred_normal.min():.4f} ~ {pred_normal.max():.4f}")
    print(f"Pred normal mean: {pred_normal.mean(dim=(1,2))}")
    
    # 计算有效像素数
    print(f"Valid mask sum: {valid_mask.sum()} / {valid_mask.numel()}")
    
    # 计算 cosine similarity 的分布
    cosine_sim = (F.normalize(pred_normal, dim=0) * 
                  F.normalize(gt_normal, dim=0)).sum(dim=0)
    print(f"Cosine similarity: mean={cosine_sim[valid_mask].mean():.4f}, "
          f"std={cosine_sim[valid_mask].std():.4f}")
    print(f"Loss mono normal: {loss_mono_normal:.6f}")
```

### 视觉验证

```python
# 在 tensorboard 中添加可视化
if tb_writer and iteration == loss_scheduler.stages.stage2_end:
    # 可视化 GT 法线
    gt_normal_vis = (gt_normal * 0.5 + 0.5).clamp(0, 1)
    tb_writer.add_images("debug/gt_normal", gt_normal_vis[None], iteration)
    
    # 可视化预测法线
    pred_normal_vis = (pred_normal * 0.5 + 0.5).clamp(0, 1)
    tb_writer.add_images("debug/pred_normal", pred_normal_vis[None], iteration)
    
    # 可视化 valid mask
    valid_mask_vis = valid_mask.float().unsqueeze(0)
    tb_writer.add_images("debug/valid_mask", valid_mask_vis[None], iteration)
```

---

## 结论

**为什么训练效果不好？**

主要原因是 **坐标系不一致** 和 **法线变换公式错误**：

1. GT 法线在一个坐标系（Open3D），pred_normal 在另一个坐标系（COLMAP 的变换后）
2. 法线变换使用了错误的矩阵（W2C.T 而不是 C2W.T）
3. 这导致 gt_norm 和 pred_norm 在不同的基准上计算 cosine similarity，结果无意义

**修复后期望效果**：
- Normal supervision loss 会显著下降
- 几何结构会更加清晰
- 相机法向一致性会大幅改善

