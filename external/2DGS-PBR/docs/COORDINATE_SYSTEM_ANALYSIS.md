# 2DGS-PBR 坐标系问题详细分析

## 问题现象
1. `m_n = 0.05` 几乎不变（cosine_sim ≈ 0，法线正交）
2. `m_d = nan`
3. GT 法线和渲染法线似乎在不同坐标系

## 1. 2DGS 的 world_view_transform 矩阵分析

### 1.1 getWorld2View2 计算流程（scene/cameras.py:61）

```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
```

### 1.2 getWorld2View2 实现（utils/graphics_utils.py:38-49）

```python
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()     # R is stored transposed
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)        # Rt is W2C, invert to get C2W
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)        # Re-invert to get W2C
    return np.float32(Rt)
```

### 1.3 矩阵流转分析

**输入数据来自 dataset_readers.py (readCamerasFromTransforms):**

```
c2w = np.array(frame["transform_matrix"])  # OpenGL/Blender: X右, Y上, Z后
# c2w[:3, 1:3] *= -1  # 已被注释（原意转换到COLMAP坐标系）

w2c = np.linalg.inv(c2w)
R = np.transpose(w2c[:3,:3])  # R存储时被转置
T = w2c[:3, 3]
```

**getWorld2View2中的处理:**

1. `Rt[:3, :3] = R.transpose()` → 再次转置回 w2c[:3,:3]
2. `Rt[:3, 3] = T` → w2c的平移向量
3. **结论：返回的是标准的 W2C 矩阵**

**在Camera中的使用：**

```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, ...)).transpose(0, 1).cuda()
```

- getWorld2View2 返回 W2C 矩阵 (4x4)
- `.transpose(0, 1)` 将其转置
- **结论：world_view_transform 是 W2C 的转置形式，即行向量矩阵**

---

## 2. 法线变换分析

### 2.1 rasterizer 输出的法线（gaussian_renderer/__init__.py:184）

```python
render_normal = allmap[2:5]  # [3, H, W]
```

- **坐标系：view space（相机空间）**
- 来自rasterizer的直接输出，在camera space中计算

### 2.2 法线变换（gaussian_renderer/__init__.py:185）

```python
# world_view_transform is stored transposed, so [:3,:3] is already R_w2c^T (i.e., R_v2w)
render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3])).permute(2,0,1)
```

**数学分析：**

- `render_normal` shape: [3, H, W] → permute → [H, W, 3]
- `world_view_transform[:3,:3]` shape: [3, 3]

**矩阵关系：**
- `world_view_transform` 是 W2C 的转置
- `world_view_transform[:3,:3]` 是 R_w2c^T = R_c2w（但这里的说法有问题）

让我重新分析：

假设 `world_view_transform` = [[R_w2c^T, T], [0,0,0,1]]

则 `world_view_transform[:3,:3]` = R_w2c^T

对于法线变换：
- 如果 n_view 在view space，要变到world space
- 需要 n_world = R_c2w @ n_view = R_w2c^T @ n_view

**结论：变换数学上正确，但需要验证view space法线的方向**

---

## 3. GT 法线生成分析

### 3.1 Open3D渲染（scripts/render_gt_open3d.py:162-172）

```python
mesh_normal = o3d.geometry.TriangleMesh(mesh)
normals = np.asarray(mesh_normal.vertex_normals)  # 世界坐标系

# Map [-1, 1] -> [0, 1]
colors = (normals + 1.0) * 0.5
mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)
```

**关键点：**
- `mesh.vertex_normals` 是 **世界坐标系** 的法线
- 存储为顶点颜色，在Unlit着色器下直接输出

### 3.2 坐标系变换（scripts/render_gt_open3d.py:215-222）

```python
flip_yz = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])
c2w_o3d = c2w @ flip_yz
w2c = np.linalg.inv(c2w_o3d)

renderer.setup_camera(intrinsic, w2c)
```

**关键问题：**

**flip_yz 作用于相机姿态，不作用于网格顶点！**

因此：
1. 网格顶点仍在原始坐标系（OpenGL/Blender：X右, Y上, Z后）
2. flip_yz 只改变了相机的观察方向
3. 顶点颜色中的世界坐标系法线 **未被 flip_yz 变换**

**这是第一个大问题！**

### 3.3 Open3D 中的渲染

- OffscreenRenderer 在 Open3D 坐标系中渲染（Y下, Z前）
- 但顶点颜色 = 世界坐标系法线，**没有经过坐标系转换**

---

## 4. 深度分析

### 4.1 2DGS 深度（gaussian_renderer/__init__.py:192-203）

```python
render_depth_expected = allmap[0:1]
render_depth_median = allmap[5:6]

surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
```

- **坐标系：view space（相机空间）**
- **单位：米（绝对深度）**

### 4.2 GT 深度（scripts/render_gt_open3d.py:248-249）

```python
depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
depth_np = np.asarray(depth_o3d)
```

- `z_in_view_space=True`：返回 **view space深度**
- **单位：米（绝对深度）**
- **坐标系转换影响：**
  - flip_yz 改变了相机空间的Z轴指向
  - 但深度值本身（z坐标的绝对值）应该保持一致
  - **可能的问题：flip_yz 后Z方向反向，深度符号可能反转**

---

## 5. 一致性检查

### 5.1 GT 法线坐标系

输出的 GT 法线是什么坐标系？

让我们追踪：
1. mesh.vertex_normals → 世界坐标系（假设X右, Y上, Z后）
2. 存储为顶点颜色，在Unlit着色下直接渲染
3. Open3D renderer with w2c 矩阵

**Open3D 渲染流程：**
- 输入：顶点位置和颜色在世界坐标系
- 顶点颜色 = 法线（世界坐标）
- 相机 setup_camera(w2c)：应用 w2c 变换到顶点，但**颜色不受变换**
- Unlit着色：直接输出颜色（法线世界坐标）

**结论：GT法线仍是世界坐标系！**

### 5.2 2DGS 渲染法线坐标系

```python
render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3])).permute(2,0,1)
```

- 输入：view space法线（allmap[2:5]）
- 乘以 R_w2c^T = R_c2w
- **结论：输出应该是世界坐标系法线**

### 5.3 坐标系失配根本原因

**dataset_readers.py 中注释掉了 flip：**

```python
# c2w[:3, 1:3] *= -1  # DISABLED: was converting to COLMAP, but GT uses OpenGL
```

这意味着：
- 训练数据使用 OpenGL 坐标系（X右, Y上, Z后）
- GT 数据也使用 OpenGL 坐标系

**但是 render_gt_open3d.py 中有 flip_yz！**

这导致：
- GT 法线仍然在 OpenGL 坐标系（未变换）
- GT 深度也在 Open3D 变换后的坐标系中

---

## 6. 根本问题总结

### 问题 A：GT 法线未正确变换

**现象：** cosine_sim ≈ 0（正交）

**原因：**
1. mesh.vertex_normals 是世界坐标法线
2. flip_yz 作用于相机，不作用于法线
3. GT 法线输出仍是原始世界坐标
4. 2DGS render_normal 也试图输出世界坐标
5. **但它们可能有符号或方向差异**

**验证方法：**
- 在 render_gt_open3d.py 中，打印一个像素的法线
- 在 2DGS 中，打印相同像素的渲染法线
- 比对是否一致

### 问题 B：GT 深度坐标系问题

**现象：** m_d = nan

**可能原因：**
1. flip_yz 改变了Z方向，depth值可能有符号问题
2. 深度对齐时 lstsq 可能遇到数值问题
3. scale_invariant_loss 中的 mask = (gt > 0) 可能排除所有深度

### 问题 C：坐标系整体不一致

**根本原因：**
1. flip_yz 应该作用于网格顶点**或**法线，但现在都没做
2. 导致 GT 法线和深度在不同的有效坐标系中

---

## 7. 修复建议

### 修复 1：改正 GT 法线（推荐）

在 `render_gt_open3d.py` 中，对法线应用坐标系变换：

```python
# 世界坐标法线
normals = np.asarray(mesh_normal.vertex_normals)

# 应用 flip_yz 到法线
flip_yz_3x3 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])
normals_flipped = normals @ flip_yz_3x3.T  # [N, 3] @ [3, 3]^T

colors = (normals_flipped + 1.0) * 0.5
mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)
```

**原因：**
- flip_yz 变换了相机坐标系，也应该变换参考坐标系的法线
- 这样 GT 法线和相机看到的法线才一致

### 修复 2：深度验证

在 `scale_invariant_loss` 之前添加检查：

```python
mask = (gt_depth > 0)
if mask.sum() == 0:
    print(f"WARNING: GT depth all invalid! Range: {gt_depth.min():.4f} ~ {gt_depth.max():.4f}")
    return torch.tensor(0.0, device="cuda")
```

这会显示深度是否有问题。

### 修复 3：验证数据一致性

在训练开始时，保存一个 debug 视图：

```python
# 在训练循环中，第一个 iteration with GT data
if iteration == loss_scheduler.stages.stage1_end:
    # 保存 GT 法线 debug visualization
    # 保存 render_normal debug visualization
    # 对比深度范围
```

---

## 8. 预期修复后的效果

- **m_n 从 ≈0 → 应升高至 0.3-0.7** （取决于相机覆盖情况）
- **m_d 从 nan → 正常数值** （≈0.1-0.5）
- 法线和深度应该有意义的监督效果

