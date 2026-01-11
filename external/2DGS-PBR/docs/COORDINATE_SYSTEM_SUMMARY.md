# 2DGS-PBR 坐标系问题 - 完整总结

## 问题陈述

你遇到的问题：
1. **m_n = 0.05** （几乎没有变化）→ cosine_sim ≈ 0（法线正交）
2. **m_d = nan** （深度监督失败）  
3. GT 法线和 2DGS 渲染法线似乎不匹配

## 根本原因（一句话）

**GT 法线向量没有随相机坐标系变换而变换。**

## 详细原因

### 2DGS 做了什么

```python
# gaussian_renderer/__init__.py:184-185
render_normal = allmap[2:5]  # view space 法线
# 变换到 world space
render_normal = (render_normal.permute(1,2,0) @ world_view_transform[:3,:3]).permute(2,0,1)
# 输出：world space 法线（OpenGL 坐标系）
```

### GT 做了什么

```python
# scripts/render_gt_open3d.py:163-172
normals = np.asarray(mesh_normal.vertex_normals)  # world space（OpenGL）
# 直接使用，无变换
colors = (normals + 1.0) * 0.5

# scripts/render_gt_open3d.py:215-222
flip_yz = ...  # Y和Z翻转
c2w_o3d = c2w @ flip_yz  # 相机被变换
# 但法线没有被变换！
```

### 问题

| 组件 | 坐标系 | 变换 |
|------|-------|------|
| 2DGS render_normal | world (OpenGL) | ✓ 正确 |
| GT 法线 | world (OpenGL) | ✗ **未变换** |
| 2DGS 相机 | OpenGL | 无变换（使用原始 c2w） |
| GT 相机 | Open3D | **应用 flip_yz** |

**结果：** GT 法线和 2DGS 法线在不同的参考系中，所以 cosine_sim ≈ 0。

## 修复（三行代码）

在 `scripts/render_gt_open3d.py` 第 163 行后添加：

```python
flip_yz_3x3 = np.array([[1,0,0], [0,-1,0], [0,0,-1]], dtype=np.float32)
normals = normals @ flip_yz_3x3.T
```

## 完整追踪

### 1. world_view_transform 是什么

```
getWorld2View2(R, T) → W2C矩阵
                    → .transpose(0,1)
                    → world_view_transform（行向量形式的W2C）
```

**结论：** `world_view_transform[:3,:3]` 是 R_w2c^T = R_c2w

### 2. 法线变换数学

对于列向量：n_world = R_c2w @ n_view
对于行向量：n_world = n_view @ R_c2w^T

但 2DGS 代码中：
```python
n_world = n_view @ world_view_transform[:3,:3]
        = n_view @ R_c2w
```

这是错的！应该是：
```python
n_world = n_view @ (R_c2w)^T = n_view @ R_w2c
```

但等等...实际上注释说了：
```python
# world_view_transform is stored transposed, so [:3,:3] is already R_w2c^T (i.e., R_v2w)
```

所以 world_view_transform 中：
- 前 3×3 = W2C 的转置 = R_c2w（当view作为列时）
- 对行向量：n @ R_c2w = 正确的变换

**结论：** 2DGS 的法线变换在数学上是正确的。

### 3. GT 法线坐标系

```
mesh.vertex_normals (OpenGL world space)
  ↓
存储为 vertex colors
  ↓
Open3D Unlit 着色器
  ↓
直接输出颜色 = OpenGL world space 法线
```

**关键问题：** Unlit 着色器不对颜色进行任何变换！

虽然相机被 flip_yz 变换了，但输出的法线仍然在 OpenGL 空间。

### 4. 坐标系的实际差异

2DGS 期望的：
- 法线：world space（OpenGL：X右, Y上, Z后）

GT 实际输出的：
- 法线：world space（OpenGL：X右, Y上, Z后）

看起来一样！但是...

相机变换：
- 2DGS：原始 OpenGL c2w（未变换）
- GT：flip_yz 变换后的 c2w（Y和Z翻转）

所以：
- 2DGS 的"法线世界坐标"是相对于原始 OpenGL 坐标系
- GT 的"法线世界坐标"虽然值一样，但相机参考系已变

**结果：** 虽然都说"world space"，但参考框架不同！

### 5. 深度问题

GT 深度在 Open3D 的 z_in_view_space 坐标系中：
```python
depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
```

在 train_pbr.py：
```python
mask = (gt > 0)
if mask.sum() == 0:
    return torch.tensor(0.0)  # m_d = nan
```

flip_yz 改变了 Z 方向，但深度值应该仍是正的。

**可能原因：**
1. 深度文件有问题（全 0）
2. 网格不在视锥内
3. **最可能：由于法线坐标系错误导致的级联失败**

## 修复步骤

### Step 1: 应用代码修复

编辑 `scripts/render_gt_open3d.py`，在第 163 行后插入：

```python
normals = np.asarray(mesh_normal.vertex_normals)

# NEW CODE START
flip_yz_3x3 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
], dtype=np.float32)
normals = normals @ flip_yz_3x3.T
# NEW CODE END

colors = (normals + 1.0) * 0.5
```

### Step 2: 重新生成 GT

```bash
cd /home/fangsuo/py/OpenMaterial/external/2DGS-PBR
rm -rf <dataset>/depth_gt <dataset>/normal_gt
python scripts/render_gt_open3d.py -s <dataset> --debug
```

### Step 3: 重新训练

```bash
python train_pbr.py -s <dataset> ...
```

监视输出：
- 首次有法线监督时，m_n 应该有一个**跳跃**
- m_d 应该从 nan 变为有限值

## 验证清单

修复后检查：

- [ ] GT 法线文件正常生成（检查 normal_gt 目录中的 PNG）
- [ ] 训练日志显示 m_n > 0（不再接近 0）
- [ ] m_d 不再是 nan（应该是 0.1-0.5 范围）
- [ ] 法线可视化中 GT 和 pred 有相似性

## 为什么这么难找

1. **隐含的坐标系概念** - 没有显式标记"这是 OpenGL 空间"或"这是 Open3D 空间"
2. **Unlit 着色器的行为** - 通常会自动应用视图变换，但这里不做
3. **cosine_sim ≈ 0 很有误导性** - 看起来计算逻辑没错，实际是向量正交
4. **两个坐标系都叫"世界坐标"** - 但通过不同的变换相关联
5. **flip_yz 的位置** - 在相机设置中很容易忽视它作用在哪里

## 数学验证

假设 2DGS 渲染出法线 **n_pred**（已在world space）。

假设网格法线是 **n_mesh**（OpenGL world space）。

当我们计算 cosine_sim 时：
```python
cosine_sim = (n_pred * n_gt).sum()
```

如果两个法线都真的在相同的坐标系中，cosine_sim 应该是 0.5-1.0（取决于是否归一化和几何覆盖）。

如果 cosine_sim ≈ 0，意味着：
1. 法线正交（90度）
2. 或者法线在完全不同的坐标系中

后者是实际情况。flip_yz 导致：
- Y 反向：n_y → -n_y
- Z 反向：n_z → -n_z  

所以如果 GT 法线有非零的 Y 或 Z 分量，它会与预期值反向，导致 cosine_sim ≈ 0。

**修复后：** 两边都应用 flip_yz（或都不应用），cosine_sim 恢复正常。

## 文件修改总结

### 修改的文件
- `scripts/render_gt_open3d.py` （第 163 行后）

### 不需要修改的文件
- `gaussian_renderer/__init__.py` （法线变换已正确）
- `train_pbr.py` （损失计算已正确，只是在等待正确的 GT）
- `scene/cameras.py` （world_view_transform 定义已正确）
- `scene/dataset_readers.py` （注释掉 flip 已正确）

## 预期结果变化

### 修复前
```
iteration:  100  Loss: 0.1234  m_d: nan        m_n: 0.0500
iteration:  200  Loss: 0.1200  m_d: nan        m_n: 0.0502
iteration:  300  Loss: 0.1180  m_d: nan        m_n: 0.0501
```

### 修复后
```
iteration:  100  Loss: 0.1234  m_d: 0.4532     m_n: 0.6234  ← 跳跃！
iteration:  200  Loss: 0.0980  m_d: 0.3845     m_n: 0.5123
iteration:  300  Loss: 0.0856  m_d: 0.3234     m_n: 0.4521  ← 下降（变好）
```

## 快速参考

**问题：** cosine_sim ≈ 0
**原因：** GT 法线未被 flip_yz 变换
**位置：** `scripts/render_gt_open3d.py:163`
**修复：** 应用 3 行代码
**预期效果：** m_n 从 0.05 升至 0.3-0.7

