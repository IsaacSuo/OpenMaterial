# Mono Normal Supervision 修复指南

## 问题总结

2DGS-PBR 的 Mono Normal Supervision 训练效果差的根本原因：

1. **坐标系不一致**：GT 法线在 Open3D 坐标系，但 2DGS 使用 COLMAP 坐标系变换
2. **法线变换公式错误**：使用 W2C.T 而非正确的 C2W.T
3. **背景检测不鲁棒**：L1 范数阈值检测不够精确

---

## 修复步骤

### 步骤 1: 修复 render_gt_open3d.py

**目的**：统一使用 OpenGL 坐标系，避免 flip_yz 导致的坐标系混淆

**文件**：`scripts/render_gt_open3d.py`

**修改位置**：第 211-234 行

**修改前**：
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

        # Apply flip to camera-to-world, then invert to get world-to-camera
        c2w_o3d = c2w @ flip_yz
        w2c = np.linalg.inv(c2w_o3d)

        if debug:
            cam_pos = c2w[:3, 3]
            print(f"[DEBUG] Camera '{cam['name']}': pos={cam_pos}, W={W}, H={H}")
            print(f"[DEBUG] c2w:\n{c2w}")
            print(f"[DEBUG] w2c (after flip):\n{w2c}")

        # Setup camera in renderer
        # Note: OffscreenRenderer.setup_camera takes (intrinsic, extrinsic_matrix)
        renderer.setup_camera(intrinsic, w2c)
```

**修改后**：
```python
        # Note: Open3D and OpenGL both work with the same coordinate convention
        # when properly set up. We directly use the c2w without flip_yz
        # since the mesh is already in the correct world coordinate system.
        
        # Get world-to-camera transform (required by Open3D)
        w2c = np.linalg.inv(c2w)

        if debug:
            cam_pos = c2w[:3, 3]
            print(f"[DEBUG] Camera '{cam['name']}': pos={cam_pos}, W={W}, H={H}")
            print(f"[DEBUG] c2w:\n{c2w}")
            print(f"[DEBUG] w2c:\n{w2c}")

        # Setup camera in renderer
        # Note: OffscreenRenderer.setup_camera takes (intrinsic, extrinsic_matrix)
        renderer.setup_camera(intrinsic, w2c)
```

**理由**：
- Open3D 的 OffscreenRenderer 会自动处理坐标系转换
- mesh 的法线已经在世界坐标系中，不需要额外 flip
- 去除 flip_yz 确保 GT 法线坐标系与后续 2DGS 处理一致

---

### 步骤 2: 修复 scene/dataset_readers.py

**目的**：保持一致的坐标系，不混合 OpenGL 和 COLMAP 坐标系

**文件**：`scene/dataset_readers.py`

**修改位置**：第 194-202 行

**修改前**：
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

**修改后**：
```python
        # NeRF 'transform_matrix' is a camera-to-world transform (OpenGL convention)
        c2w = np.array(frame["transform_matrix"])
        # Keep OpenGL convention consistent with GT normal generation
        # (do not flip Y and Z axes - let the rasterizer handle the coordinate system)

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
```

**理由**：
- 移除 `c2w[:3, 1:3] *= -1` 坐标系转换
- 保持与 render_gt_open3d.py 一致的坐标系
- CUDA rasterizer 应该能正确处理 OpenGL 坐标系

---

### 步骤 3: 修复 gaussian_renderer/__init__.py

**目的**：使用正确的法线变换公式（W2C 的逆转置而不是转置）

**文件**：`gaussian_renderer/__init__.py`

**修改位置**：第 183-184 行

**修改前**：
```python
    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
```

**修改后**：
```python
    # get normal map
    # transform normal from view space to world space
    # Note: normals are direction vectors, need inverse-transpose transformation
    render_normal = allmap[2:5]
    W2C_rot = viewpoint_camera.world_view_transform[:3,:3]
    # For normal transformation: n_world = (W2C^-T) @ n_view = (C2W^T) @ n_view
    C2W_rot = torch.inverse(W2C_rot)
    render_normal = (render_normal.permute(1,2,0) @ C2W_rot.T).permute(2,0,1)
```

**理由**：
- 法线是方向向量，需要用逆转置变换而不是简单转置
- 正确公式：`n_world = (W2C^-T) @ n_view`
- 这确保法线方向正确变换到世界坐标系

**数学验证**：
```
如果 M 是位置变换矩阵（从坐标系 A 到 B）：
  p_B = M @ p_A

那么法线的正确变换是（注意区别）：
  n_B = (M^-T) @ n_A = (M^-1)^T @ n_A

对于我们的情况：
  M = W2C（从世界到视图）
  n_view = 视图空间法线
  n_world = (W2C^-T) @ n_view = (C2W)^T @ n_view ✓
```

---

### 步骤 4: 改进 train_pbr.py 的 valid_mask

**目的**：使用更鲁棒的背景检测方法

**文件**：`train_pbr.py`

**修改位置**：第 295-301 行

**修改前**：
```python
                with profiler.profile("mono_normal_loss"):
                    pred_norm = F.normalize(pred_normal, dim=0)
                    gt_norm = F.normalize(gt_normal, dim=0)
                    cosine_sim = (pred_norm * gt_norm).sum(dim=0)
                    valid_mask = (gt_normal.abs().sum(dim=0) > 0.1)

                    if valid_mask.sum() > 0:
                        loss_mono_normal = weights['mono_normal'] * (1.0 - cosine_sim[valid_mask]).mean()
```

**修改后**：
```python
                with profiler.profile("mono_normal_loss"):
                    pred_norm = F.normalize(pred_normal, dim=0)
                    gt_norm = F.normalize(gt_normal, dim=0)
                    cosine_sim = (pred_norm * gt_norm).sum(dim=0)
                    
                    # Use L2 norm for more robust background detection
                    # Background pixels have near-zero normal magnitude
                    valid_mask = (torch.norm(gt_normal, dim=0) > 0.1)
                    
                    # Additionally filter by confidence (if alpha map available)
                    # valid_mask = valid_mask & (render_pkg.get('rend_alpha', torch.ones_like(valid_mask))[0] > 0.5)

                    if valid_mask.sum() > 0:
                        loss_mono_normal = weights['mono_normal'] * (1.0 - cosine_sim[valid_mask]).mean()
                    else:
                        loss_mono_normal = torch.tensor(0.0, device="cuda")
```

**理由**：
- L2 范数（欧几里得范数）比 L1 范数更符合几何直观
- 添加了安全检查（如果没有有效像素则返回 0）
- 注释了可选的 alpha map 过滤（如需要可启用）

---

## 实施步骤

### 第 1 阶段：代码修改

1. **备份原始文件**：
```bash
cd /home/fangsuo/py/OpenMaterial/external/2DGS-PBR
cp scripts/render_gt_open3d.py scripts/render_gt_open3d.py.bak
cp scene/dataset_readers.py scene/dataset_readers.py.bak
cp gaussian_renderer/__init__.py gaussian_renderer/__init__.py.bak
cp train_pbr.py train_pbr.py.bak
```

2. **应用修改**：
   - 按步骤 1-4 修改相应的文件

3. **验证语法**：
```bash
python -m py_compile scripts/render_gt_open3d.py
python -m py_compile scene/dataset_readers.py
python -m py_compile gaussian_renderer/__init__.py
python -m py_compile train_pbr.py
```

### 第 2 阶段：重新生成 GT 法线

**重新渲染 GT 法线**（使用修复后的脚本）：

```bash
python scripts/render_gt_open3d.py \
    -s path/to/your/dataset \
    -g path/to/gt_mesh.ply \
    --debug
```

检查输出的 `normal_gt/` 目录中的法线是否正确。

### 第 3 阶段：从零开始训练

**清空旧的训练输出**：
```bash
rm -rf output/*
```

**使用修复后的代码训练**：
```bash
python train_pbr.py \
    -s path/to/your/dataset \
    --use_pseudo_gt \
    --normal_subdir normal_gt \
    --lambda_mono_normal 0.05 \
    --stage1_end 3000 \
    --stage2_end 7000
```

### 第 4 阶段：监控训练

**添加调试输出**（可选）：

在 `train_pbr.py` 中 mono normal loss 计算后添加：

```python
                    # Debug logging (comment out after verification)
                    if iteration % 100 == 0:
                        print(f"[Iter {iteration}] Mono Normal Loss Debug:")
                        print(f"  Valid pixels: {valid_mask.sum().item()} / {valid_mask.numel()}")
                        print(f"  Cosine sim: mean={cosine_sim[valid_mask].mean():.4f}, "
                              f"std={cosine_sim[valid_mask].std():.4f}")
                        print(f"  Loss: {loss_mono_normal.item():.6f}")
```

---

## 预期改进

### 训练指标
- **Mono Normal Loss**：应该从初期的较高值逐步下降
- **PSNR**：几何约束应该改善渲染质量
- **法线一致性**：渲染法线与 GT 法线的相似度应该增加

### 质量指标
- **几何清晰度**：物体边界和细节更清晰
- **法向连贯性**：曲面法线变化更平滑
- **避免伪影**：减少由于坐标系混淆导致的奇怪法线翻转

---

## 故障排查

### 问题 1：修改后 loss 变得很大

**可能原因**：
- 坐标系仍然不匹配
- 法线方向完全反转

**解决方案**：
```python
# 在 train_pbr.py 中添加调试
if iteration == loss_scheduler.stages.stage2_end + 10:
    # 可视化 GT 和预测的法线
    print(f"GT normal sample: {gt_normal[:, 100, 100]}")
    print(f"Pred normal sample: {pred_normal[:, 100, 100]}")
    print(f"Cosine similarity: {cosine_sim[100, 100]}")
```

如果 cosine_sim 接近 -1（180 度反向），说明需要调整符号。

### 问题 2：修改后 loss 不下降

**可能原因**：
- valid_mask 过度过滤
- 法线初始化问题

**解决方案**：
```python
# 检查有效像素数
print(f"Valid pixels ratio: {valid_mask.sum() / valid_mask.numel():.2%}")
# 应该是 20-80% 的像素

# 检查 GT 法线的统计
print(f"GT normal mean: {gt_normal.mean(dim=(1,2))}")
# 应该接近 (0, 0, 0)
```

### 问题 3：与旧的训练结果不兼容

**解决方案**：
- 必须从零开始训练（不能加载旧的 checkpoint）
- 旧的 checkpoint 是在错误的坐标系下训练的

---

## 测试计划

### 单元测试（推荐添加）

```python
# 在 test_pbr_shading.py 中添加

def test_normal_transformation():
    """验证法线变换的正确性"""
    import torch
    
    # 创建测试用例
    # 视图空间法线（指向上方）
    n_view = torch.tensor([0.0, 0.0, 1.0])
    
    # 假设世界坐标系和视图坐标系相同
    # （恒等变换）
    W2C = torch.eye(3)
    C2W = torch.inverse(W2C)
    
    # 应用变换
    n_world = (n_view.unsqueeze(0) @ C2W.T).squeeze(0)
    
    # 验证：应该还是 [0, 0, 1]
    assert torch.allclose(n_world, n_view, atol=1e-5)
    print("✓ Normal transformation test passed")

def test_valid_mask():
    """验证 valid_mask 的正确性"""
    import torch
    
    # 背景像素（PNG 值 128, 128, 128）
    bg_normal = torch.tensor([0.003, 0.003, 0.003])
    
    # 前景像素（单位向量）
    fg_normal = torch.tensor([0.577, 0.577, 0.577])  # (1, 1, 1) / sqrt(3)
    
    # 应用阈值
    bg_valid = torch.norm(bg_normal) > 0.1  # False
    fg_valid = torch.norm(fg_normal) > 0.1  # True
    
    assert not bg_valid
    assert fg_valid
    print("✓ Valid mask test passed")
```

---

## 提交检查清单

- [ ] 所有 4 个文件已修改
- [ ] 语法检查通过
- [ ] 重新生成了 GT 法线
- [ ] 从零开始训练
- [ ] 监控了至少 1 个 epoch 的训练
- [ ] 法线 loss 在下降
- [ ] 可视化了 GT 和预测法线
- [ ] 没有发现奇怪的伪影或异常

---

## 相关文档

- 详细分析：`MONO_NORMAL_SUPERVISION_ANALYSIS.md`
- 坐标系文档：见下节

---

## 坐标系参考表

| 系统 | X 轴 | Y 轴 | Z 轴 | 相机朝向 |
|------|------|------|------|---------|
| OpenGL | 右 | 上 | 后 | -Z |
| Open3D | 右 | 下 | 前 | +Z |
| COLMAP | 右 | 下 | 前 | +Z |

修复后应该统一使用 **OpenGL** 或 **Open3D/COLMAP**（两者可通过简单矩阵变换相互转换）。

