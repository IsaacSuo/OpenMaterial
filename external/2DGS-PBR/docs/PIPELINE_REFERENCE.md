# 2DGS-PBR Pipeline Reference (Detailed)

这份文档面向需要“读懂并能改代码”的使用者，按 **数据/场景 → 2DGS 可微光栅化 → PBR（G-buffer→着色→合成）→ 训练/渲染脚本** 的顺序，把 `external/2DGS-PBR` 的关键模块、张量约定、文件格式与常见坑完整记录下来。

> 说明：本文档以当前仓库 `external/2DGS-PBR` 的代码为准；很多实现源自 GraphDECO 3DGS/2DGS 体系，但本文只陈述仓库里实际存在的行为与接口。

---

<a id="sec-0-mental-model"></a>
## 0. 读代码的“全局心智模型”

**核心对象**

- `scene.cameras.Camera`：每一帧相机（外参/内参派生的 FoV、投影矩阵、GT 图像、可选 GT mask）。
- `scene.gaussian_model.GaussianModel`：所有 surfels/gaussians 的可学习参数（几何、外观、PBR 材质）。
- `gaussian_renderer.render()`：把 `GaussianModel` 在某个 `Camera` 下光栅化为图像 + 辅助缓冲（alpha、normal、depth、distortion…），并可选输出 PBR G-buffer（albedo/roughness/metallic）。
- `utils.pbr_utils`：环境光 `EnvironmentLight` 与屏幕空间 PBR 着色（IBL + Cook-Torrance 近似）。

**两条主要脚本链路**

- “原始 2DGS 训练”（SH/RGB 监督 + normal/distortion 正则）：`train.py`
- “PBR 静态几何训练”（只优化材质+光照，锁几何/禁 densification）：`train_pbr.py`（函数名 `training_pbr_static`）

**PBR 的关键约定：G-buffer 是 premultiplied 的**

`gaussian_renderer.render(render_pbr=True)` 输出的
`gbuffer_*` 与 `rend_normal` 都是**按 `rend_alpha` 预乘**的（premultiplied）。要获得物理意义正确的材质/法线，应使用：

`unpremult = premult / (alpha + eps)`  

对应实现见：

- `train_pbr.py`：训练时会对 `gbuffer_albedo/roughness/metallic` 和 `rend_normal` 统一反预乘
- `render_pbr.py`：渲染导出也会反预乘后保存材质贴图与 PBR shaded

---

<a id="sec-1-repo-map"></a>
## 1. 目录与模块职责（Repo Map）

### 1.1 训练/渲染脚本（entrypoints）

- `train.py`：基础 2DGS 训练（SH/RGB 重建 + distortion/normal 正则 + densification）。
- `train_pbr.py`：PBR 静态几何训练（dense 点云初始化、锁 xyz/rotation、禁 densification、学材质+env light、对“物体+skybox 合成”监督）。
- `train_env_light.py`：可选的环境光预训练/初始化脚本（从多视角监督学习一个 env_light 初值，供 `train_pbr.py` 继续优化并用 prior 锚定）。
- `render.py`：基础渲染/导出 + TSDF mesh 抽取（依赖 `open3d`、`utils/mesh_utils.py` 等）。
- `render_pbr.py`：PBR 渲染导出（输出合成图、材质贴图、normal/depth 可视化，并可计算指标）。
- `view.py` + `gaussian_renderer/network_gui.py`：交互式 viewer（socket 协议传输相机与渲染结果）。

### 1.2 数据与场景

- `arguments/__init__.py`：
  - `ModelParams`：数据集路径、图像目录、分辨率缩放、白背景、render_items 等。
  - `PipelineParams`：渲染/几何相关（如 `depth_ratio`、`compute_cov3D_python`）。
  - `OptimizationParams`：训练超参（iterations、各种 lr、densification/正则权重等）。
  - `get_combined_args()`：从 `--model_path/cfg_args` 读取历史配置并与命令行合并（渲染脚本常用）。
- `scene/__init__.py:Scene`：
  - 根据 `source_path` 自动判断数据集格式：COLMAP（`sparse/`）或 Blender（`transforms_*.json`）。
  - 构建 train/test 相机列表，并计算 `cameras_extent`（用于学习率缩放/一些阈值）。
  - 管理模型加载/保存（`point_cloud/iteration_x/point_cloud.ply`）。
- `scene/dataset_readers.py`：
  - COLMAP / Blender 数据读取。
  - `fetchPly()`：读取 `.ply` 点云为 `BasicPointCloud(points, colors, normals)`。
  - Blender transforms 路径支持可选 `mask`：优先尝试从 `/mask/` 目录读取 mask；否则可从 RGBA alpha 提取（并在 alpha 来自图像通道时对 RGB 做背景填充）。
- `utils/camera_utils.py`：
  - `loadCam()`：把 `CameraInfo`（PIL 图）转成 `scene.cameras.Camera`（torch tensor），并对 mask 做对齐与下采样。
- `scripts/reconstruct_ground_plane_texture.py`：
  - 用“地面平面 + 多视角重投影一致性”从背景像素拼接平面纹理（例如棋盘格地面）。

### 1.3 可微光栅化与输出缓冲

- `gaussian_renderer/__init__.py`：
  - 组装 `GaussianRasterizationSettings`（FoV、投影矩阵、bg、campos 等）。
  - 调用 submodule `diff_surfel_rasterization` 的 CUDA 扩展执行光栅化。
  - 解析返回的 `allmap`/`out_others`：alpha、normal、depth、distortion、median depth 等。
  - 若 `render_pbr=True` 且 `pc.use_pbr=True`：额外输出 PBR G-buffer。

### 1.4 PBR（着色/环境光/损失）

- `utils/pbr_utils.py`：
  - `EnvironmentLight`：可学习 equirectangular env map（形状 `[3, H, 2H]`），包含：
    - solid-angle 权重 `sin(theta)`（用于 TV/平滑正则，避免极区像素密度不均导致的偏置）
    - `sample()`：按方向采样 env map
    - `sample_prefiltered()`：粗糙度驱动 mipmap 预滤波采样（用于 IBL specular/ diffuse irradiance 近似）
    - `tv_loss_weighted()` / `smoothness_loss_weighted()` / `register_gradient_scaling_hook()`
  - `GroundPlane`：可选的“有限深度背景”（例如棋盘格地面），用于替代纯 skybox：
    - `sample(ray_dirs_world, camera_center)`：返回 `(ground_color, hit_mask)`，其中 `ground_color` 在未命中像素为 0，`hit_mask` 指示哪些像素射线与平面相交且落在 UV bounds 内
    - `train_pbr.py` / `render_pbr.py` 中会把 ground 与 skybox 按 `hit_mask` 合成背景
  - `screen_space_pbr_shading()`：输入 G-buffer（albedo/roughness/metallic/normal/depth）与视线方向（可传入 `ray_dirs_world`），输出 `[3,H,W]` PBR shaded。
  - `compute_ray_directions_world_from_fov()`：根据 FoV + `Camera.world_view_transform` 计算每像素世界空间射线方向（注意本 repo 的 transform 约定见 §4）。
- `utils/loss_utils.py`：
  - `l1_loss()` 支持 mask（实现为加权平均）。
  - `ssim()` 支持 mask（当 mask 是单通道时会扩展到多通道避免 SSIM>1 的异常归一化）。
  - `compute_pbr_losses()`：材质图 TV + prior + chroma consistency，返回字典（含 `total_pbr_reg`）。
- `utils/image_utils.py`：
  - `psnr()` 支持 mask（会把单通道 mask 扩展到 RGB 通道，确保 masked PSNR 的归一化与 unmasked 一致）。

---

<a id="sec-2-conventions"></a>
## 2. 张量/形状/坐标约定（最易出错的地方）

### 2.1 图像张量的通道顺序与形状

本 repo 大多数渲染与损失以 **`[C, H, W]`** 的 torch tensor 表示图像（例如 `render`、`original_image`、`gbuffer_*`、`rend_alpha`）。

少数函数内部会转为 **`[H, W, C]`** 便于按像素做向量运算（例如 `screen_space_pbr_shading()` 内部会 `permute`）。

### 2.2 PBR G-buffer 的 premultiply / unpremultiply

`gaussian_renderer.render(render_pbr=True)` 的 G-buffer 渲染方式是：把每个 gaussian 的属性（albedo/roughness/metallic）当作 “color” 送入同一个 rasterizer，与 opacity/alpha 一起做加权累积。因此输出相当于：

- `gbuffer_*_pm ≈ Σ (attr_i * w_i)`  
- `alpha ≈ Σ (w_i)`（更准确是透射累积后 `1-T`）

要得到物理意义正确的属性图，需要反预乘：

- `attr = clamp(gbuffer_attr_pm / (alpha + eps))`
- `normal = (rend_normal_pm / (alpha + eps))`（后续通常再 normalize）

对应实现：

- `train_pbr.py`：训练中对 albedo/roughness/metallic/normal 都做了反预乘
- `render_pbr.py`：导出时也做了反预乘再保存材质贴图与法线可视化

**注意**：如果你直接对 premultiplied G-buffer 做正则（TV/prior），等价于把背景（alpha≈0）区域也引入数值不稳定；因此训练代码用 `reg_mask = mask if mask else alpha.detach()` 来限制正则作用范围。

### 2.3 `render()` 返回包的字段（关键字段）

在 `gaussian_renderer/__init__.py:render()` 中，常用字段如下：

- `render`：SH/RGB 渲染结果 `[3,H,W]`
- `rend_alpha`：alpha `[1,H,W]`
- `rend_normal`：世界空间法线（但仍 premultiplied）`[3,H,W]`
- `surf_depth`：用于几何正则/后处理的“伪表面深度” `[1,H,W]`（由 expected/median depth 混合得到，受 `pipe.depth_ratio` 控制）
- `surf_normal`：由 `depth_to_normal` 从 `surf_depth` 求出的伪表面法线（用于 normal consistency 正则），并乘了 `rend_alpha.detach()`
- `rend_dist`：distortion 辅助图（用于 depth distortion 正则）`[1,H,W]`
- （PBR）`gbuffer_albedo` / `gbuffer_roughness` / `gbuffer_metallic`：均为 premultiplied 输出

### 2.4 CUDA rasterizer 的 `out_others` 通道布局（来源于 submodule）

submodule `submodules/diff-surfel-rasterization` 的 forward 在 CUDA 中写 `out_others`（形状 `[7,H,W]`，由 `3+3+1` 以及 offsets 定义）：

- `DEPTH_OFFSET = 0`：expected depth
- `ALPHA_OFFSET = 1`：alpha（代码写 `1 - T`）
- `NORMAL_OFFSET = 2..4`：normal xyz（view-space 累积后在 Python 里转 world-space）
- `MIDDEPTH_OFFSET = 5`：median depth
- `DISTORTION_OFFSET = 6`：distortion

定义位置：`submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h`

---

<a id="sec-3-data-formats"></a>
## 3. 数据集与文件格式

### 3.1 COLMAP 数据集

`Scene` 判断 `source_path/sparse` 存在即走 COLMAP 读入（`scene/dataset_readers.py:readColmapSceneInfo`）。

期望目录结构（典型）：

- `source_path/images/`：图像
- `source_path/sparse/0/`：COLMAP 输出（`images.bin|txt`, `cameras.bin|txt`, `points3D.bin|txt`）
- `points3D.ply`：若不存在，会从 points3D.bin/txt 转一次并写到 `sparse/0/points3D.ply`

### 3.2 Blender/NeRF Synthetic 数据集

`Scene` 判断 `source_path/transforms_train.json` 存在即走 Blender。

mask 读取策略（`scene/dataset_readers.py:readCamerasFromTransforms` + `utils/camera_utils.py:loadCam`）：

1. 若存在 sibling mask 文件：把 `.../images/...` 替换为 `.../mask/...` 读取（模式 `L`）。
2. 否则若源图是 RGBA/LA：从 alpha 通道提取 mask。
3. 如果 mask 来自图像 alpha 通道，为避免 object 外 RGB 未定义导致的监督问题，会把 RGB 用白/黑背景按 alpha 合成填充（`white_background` 控制）。

最终 mask 会被转换成 torch tensor，并存入 `Camera.gt_alpha_mask`。

### 3.3 PBR 静态几何的 `--gt_ply`（dense 点云）

`train_pbr.py` 的静态几何流程要求提供 `--gt_ply`：

- 通过 `scene.dataset_readers.fetchPly()` 读取点云（支持缺少颜色/法线时的 fallback，但缺法线会影响 rotation 初始化）。
- `GaussianModel.create_from_dense_pcd()`：
  - `xyz` 来自点云点
  - `rotation` 从 normals 构造局部坐标系并转四元数
  - `scale` 通过 `simple_knn` 的 `distCUDA2` 估计邻域密度（注意代码对 NaN/inf 做了 `nan_to_num` 防护）

如果你的上游是 mesh（而不是现成的 dense 点云），本仓库提供了转换/采样脚本：

- `scripts/sample_mesh_to_ply.py`：使用 Open3D 对 mesh 做 Poisson-disk 采样，输出带法线的 dense PLY（可直接用作 `--gt_ply`）。

---

<a id="sec-4-coordinates"></a>
## 4. 坐标系与变换约定（尤其是 transform 的转置）

这套代码里有一个反复出现的约定：

- `Camera.world_view_transform` 与 `Camera.projection_matrix` 在 Python 里是 `getWorld2View2/getProjectionMatrix` 的结果**再 transpose(0,1)** 存起来的。

因此：

- 在 `utils.pbr_utils.compute_ray_directions_world_from_fov()` 中明确写了：
  - “In this repo, Camera.world_view_transform is stored transposed.”
  - 并用 `w2c = world_view_transform.transpose(0, 1)` 恢复成常规意义的 w2c。

你在新增/修改任何基于外参的几何计算时，都要优先检查：

- 你拿到的 matrix 是否已经被 transpose 存储
- 你需要的是 w2c 还是 c2w

此外，`gaussian_renderer.render()` 对 normal 做 view→world 的转换：

- rasterizer 输出 normal 在某个空间累积，Python 侧对 `allmap[2:5]` 做矩阵乘法：
  - `render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)`

这也是围绕“transform 是转置存储”的实现细节之一。

---

<a id="sec-5-gaussian-model"></a>
## 5. `GaussianModel`：可学习参数与 PBR 扩展

### 5.1 基础参数

`scene/gaussian_model.py:GaussianModel` 包含：

- `_xyz`：点位置 `[N,3]`
- `_scaling`：2DGS 的 scale（代码用 `[N,2]`，并通过 `exp` 得到正值）
- `_rotation`：四元数 `[N,4]`（`rotation_activation` 是 normalize）
- `_opacity`：参数空间，`sigmoid` 得到 opacity
- `_features_dc/_features_rest`：SH 特征（用于 SH/RGB 渲染）

### 5.2 PBR 材质参数

当 `GaussianModel(use_pbr=True)`：

- `_albedo` `[N,3]`：通过 `sigmoid` 映射到 `[0,1]`
- `_roughness` `[N,1]`：通过 `sigmoid` 后再 clamp 到 `[roughness_min, roughness_max]`
- `_metallic` `[N,1]`：通过 `sigmoid` 映射到 `[0,1]`

对应 getter：

- `get_albedo/get_roughness/get_metallic`

### 5.3 PBR 的训练模式（optimizer 组）

仓库里有多个 training_setup：

- `training_setup()`：常规训练（xyz/rotation/SH/opacity/scaling + 可选 PBR 参数）。
- `training_setup_fixed_geometry()`：锁定 `xyz/rotation`，仍训练 SH + PBR（适用于已固定几何的纹理/材质微调）。
- `training_setup_fixed_geometry_pbr_only()`：锁定 `xyz/rotation` 且不训练 SH，只训练 `scaling/opacity + PBR`（`train_pbr.py` 静态几何流程使用）。

对应 `train_pbr.py` 的“静态几何”目标：**几何来自 dense 点云，训练只解释材质与光照**。

### 5.4 PLY 保存/加载的 PBR 扩展字段

`GaussianModel.save_ply()` 会按 `construct_list_of_attributes()` 写字段；若 `use_pbr=True` 则追加：

- `albedo_0, albedo_1, albedo_2`
- `roughness`
- `metallic`

`GaussianModel.load_ply()` 若 `use_pbr=True` 且检测到 `albedo_0` 存在，会读入上述字段；否则用默认初始化。

---

<a id="sec-6-pbr-shading"></a>
## 6. PBR 着色：从 G-buffer 到合成图

### 6.1 视线方向的处理（`ray_dirs_world` 优先）

`screen_space_pbr_shading()` 支持两种 view_dir：

1. 推荐：传入每像素 `ray_dirs_world`（由 `compute_ray_directions_world_from_fov()` 计算），此时 `view_dir = normalize(-ray_dir)`。
2. fallback：不传则用常量方向近似（`-camera_center` 扩展到整图），准确性差，仅作为兼容路径。

`train_pbr.py` 与 `render_pbr.py` 都走推荐路径。

### 6.2 IBL 的 split-sum/mipmap 近似

`pbr_shading_env()` 做了两次 env 采样：

- specular：用反射方向 `reflect_dir`，并用 `roughness` 驱动 `sample_prefiltered()` 的 LOD（粗糙度越大越模糊）。
- diffuse：用 `normal` 方向采样，且 roughness 固定为 1（最大模糊）作为 irradiance 近似。

这使得粗糙度对高光形状的影响通过“采样 mip level”体现，而不是简单乘 `(1-roughness)`。

### 6.3 合成：物体 + skybox

训练与渲染都用同一合成公式：

- `bg_env = env_light.sample(ray_dirs_world)`（再 permute 成 `[3,H,W]`）
- `pred = shaded_obj * alpha + bg_env * (1 - alpha)`

并且在 PBR-only 的训练/导出里通常把 rasterizer 的 `bg_color` 设为黑色，以保证 buffer 是“干净 premultiplied”（不被背景常量颜色偏移）。

---

<a id="sec-7-train-pbr"></a>
## 7. `train_pbr.py`：静态几何 PBR 训练（逐步拆解）

> 该脚本不是对 `train.py` 的简单加权，而是一个明确的“固定几何、只学材质+环境光”的训练流程。

### 7.1 初始化阶段

1. 校验 `--gt_ply` 必须存在。
2. 读取 dense PLY：`fetchPly(gt_ply)` → `BasicPointCloud(points/colors/normals)`。
3. 初始化 `GaussianModel(dataset.sh_degree, use_pbr=True)`，设置 roughness clamp。
4. `gaussians.create_from_dense_pcd(pcd, spatial_lr_scale=1.0)`：
   - 注意 `spatial_lr_scale` 会在创建 `Scene` 后用 `scene.cameras_extent` 覆盖为真实值。
5. 创建 `Scene(dataset, gaussians)`：
   - 由于 `_xyz` 已非空，`Scene` 不会从 COLMAP point cloud 重新初始化 gaussians。
6. `gaussians.training_setup_fixed_geometry_pbr_only(opt)`：
   - 锁 `xyz/rotation/SH`
   - 学 `opacity/scaling + PBR(albedo/roughness/metallic)`
7. 初始化环境光 `EnvironmentLight(args.env_map, resolution=args.env_map_res)`：
   - 单独 Adam 优化器 `env_light_optimizer`，lr=`opt.env_light_lr`
   - 可选 `register_gradient_scaling_hook()`：用 solid-angle 权重缩放 env_map 梯度
   - 可选 `--env_light_pth <path>`：加载预训练/初始化的 env_light state_dict（作为初值，不是冻结）
   - 可选 `--env_light_prior_weight <w>`：启用 env prior，把 env_map 拉回初始化（减少“材质/光照互相吃解释权”的退化）
   - 可选 `--dump_env_map_on_eval`：在每次 eval 时把 `env_map` 以 `.pt` dump 到 `<model_path>/debug_env_map/`（便于定位 NaN/爆亮）
   - 可选 `--env_update_after/--env_update_interval`：控制 env_light 的更新时间表（例如后期稀疏更新）
   - 可选 `--batch_cams <B>`：每个 iteration 随机取 B 个相机做梯度累积（对非朗伯、高光/镜面更稳定；吞吐会近似下降到 1/B）
8. 可选初始化 GroundPlane（有限深度背景）：
   - `--ground_plane_json <.../ground_plane.json>`（由 `scripts/reconstruct_ground_plane_texture.py` 生成）
   - `--ground_texture <.../ground_texture.png>`（不传则默认取 json 同目录下的 `ground_texture.png`）

### 7.2 每步训练（核心）

每 iteration：

1. `gaussians.update_learning_rate(iter)`（即便锁几何，也会更新相关组的 lr；PBR-only 模式主要影响 opacity/scaling 等组）。
2. 随机取 `viewpoint_cam`（若 `--batch_cams>1`，则会取 B 个相机逐个 forward/backward，做梯度累积）。
3. 对每个相机：
   - `render(viewpoint_cam, gaussians, pipe, background, override_color=dummy_color, render_pbr=True)`：
     - `override_color` 传全 0，避免 SH 渲染进入监督（PBR-only）
   - 取 mask：`mask = viewpoint_cam.gt_alpha_mask`（若数据集提供；可通过 `--mask_binarize/--mask_dilate_px` 预处理）
   - 生成背景（skybox 或 ground+sky）：
     - `ray_dirs = compute_ray_directions_world_from_fov(...)`
     - 若未启用 ground plane：`bg = env_light.sample(ray_dirs)` → `[3,H,W]`
     - 若启用 ground plane：对每像素做 ray-plane intersection，命中则采样 `ground_texture.png`，否则采样 env_map，然后合成得到 `bg`
   - 取 alpha + G-buffer + normal/depth，反预乘：
   - `denom = alpha + eps`
   - `albedo = gbuffer_albedo_pm / denom`（clamp 到 `[0,1]`）
   - `roughness = gbuffer_roughness_pm / denom`（clamp 到 `[0.1,0.999]`）
   - `metallic = gbuffer_metallic_pm / denom`（clamp 到 `[0,1]`）
   - `normal = rend_normal_pm / denom`
   - 计算物体着色：`shaded_obj = screen_space_pbr_shading(..., env_light=env_light, ray_dirs_world=ray_dirs)`
   - 合成图监督：默认用 `rend_alpha` 合成：`pred = shaded_obj * alpha + bg * (1-alpha)`
     - 可选 `--composite_use_gt_mask`：若存在 `gt_alpha_mask`，则用 GT mask 替代 `rend_alpha` 做合成/重建（减少“opacity 抖动导致的背景监督泄漏”）
4. 重建损失的权重策略：
   - warmup：`iteration <= env_warmup_iters` 时强制 full-image supervision（让 env_map 看到背景）
   - 非 warmup：若存在 `gt_alpha_mask` 且未设置 `--supervise_background`，则重建只在 mask 内做（避免 GT 背景是黑/抠图导致 env_map 被错误监督）
   - 可选：`--lambda_pbr` 为物体区域加额外权重
5. 正则：
   - env 正则：`lambda_env_tv * env_light.tv_loss_weighted()` + 可选 `lambda_env_smooth * env_light.smoothness_loss_weighted()`
   - env prior（可选）：`--env_light_prior_weight` 让 `env_map` 贴近初始化（可选 `--env_light_prior_log_space` 在 log 空间做）
   - scale 正则：`lambda_scale_reg` 约束 gaussian 过大（在 log-scale 空间做，阈值为 `scale_reg_max_ratio * cameras_extent`）
   - alpha 监督：可选 `--lambda_alpha` 让 `rend_alpha` 接近 `gt_alpha_mask`（防止 opacity 作弊）
   - 材质正则：`opt.lambda_pbr_reg * compute_pbr_losses(...)`（warmup 阶段跳过）
6. 反传与 step：
   - 若 `--batch_cams>1`：对每个相机的 `recon/alpha/pbr_reg` 做 `backward(loss/B)`，再把 env_tv/env_prior/scale_reg 这类“与相机无关”的正则单独 backward 一次。
   - warmup：gaussians 不 step，只优化 env_light（注意 warmup 仍会尊重 `--env_update_interval` 的“每步更新”语义）
   - 非 warmup：step gaussians；env_light 按 `--env_update_after/--env_update_interval` 更新（可做后期稀疏更新）
7. clamp（可选）：
   - `env_clamp_min/max`：对 env_map 参数做硬 clamp
   - `scale_clamp_max_ratio`：对 `gaussians._scaling` 做上界 clamp（以 `cameras_extent` 比例定义）
8. 评估与保存：
   - 在 `test_iterations/save_iterations` 进行 render-eval、写 TensorBoard、保存 `point_cloud` 与 `env_light_*.pth`
   - 可选 `--debug_nonfinite_dump`：当 eval 中出现 NaN/Inf 时，将关键张量与统计信息写到 `<model_path>/debug_nonfinite/*.pt`（`--debug_nonfinite_dump_full` 会包含全分辨率 tensor；`--debug_nonfinite_raise` 会在 dump 后抛异常）
   - 若设置了 `--test_interval`，脚本会自动生成 `test_iterations = [N, 2N, ...]`（并确保包含最后一次迭代），用于周期性评测

### 7.3 早停（Early Stopping）

实现为“滑动窗口平均 loss 的相对改善”：

- 每 `early_stopping_interval` 次检查一次窗口平均
- 若相对改善不超过 `early_stopping_min_delta`，累计 `early_stopping_patience` 次后停止并保存

---

<a id="sec-8-render-pbr"></a>
## 8. `render_pbr.py`：PBR 导出与指标

流程概览：

1. `get_combined_args()`：优先加载训练目录里保存的 `cfg_args` 并与命令行合并。
2. 决定 iteration：若 `--iteration=-1` 则扫描 `point_cloud/iteration_*` 取最大。
3. 加载 `EnvironmentLight`：
   - 若存在 `env_light_{iteration}.pth` 则优先加载；否则尝试加载最新的 `env_light_*.pth`；都没有则用默认灰环境光。
4. （可选）加载 GroundPlane（有限深度背景）：
   - `--ground_plane_json` + `--ground_texture`（不传 texture 则默认 json 同目录下的 `ground_texture.png`）
5. `render_set()`：
   - `GaussianModel(use_pbr=True)` + `Scene(load_iteration=iteration)`
   - 对 train/test 相机循环渲染：
     - `render(..., render_pbr=True)` 得到 alpha + G-buffer + normal/depth
     - 背景：默认 skybox（env_map），可选 ground+sky（与 `train_pbr.py` 同逻辑）
     - 标准 SH 合成图：`render_pkg["render"] + bg*(1-alpha)`
     - PBR：反预乘 → `screen_space_pbr_shading` → 与背景合成 → 保存
   - 额外保存材质贴图（albedo/roughness/metallic）、normal 可视化（normalize 后映射到 [0,1]）、depth colormap
5. `--compute_metrics`：
   - 对 `renders/` 与 `pbr_shaded/` 分别与 `gt/` 计算 PSNR/SSIM（可选 LPIPS）

---

<a id="sec-9-train-2dgs"></a>
## 9. `train.py`：基础 2DGS 训练（与 PBR 的关系）

`train.py` 是传统 2DGS/3DGS 风格训练：

- 监督：`render_pkg["render"]` vs `gt_image` 的 L1 + DSSIM
- 正则：`rend_dist`（distortion）与 `rend_normal` vs `surf_normal`（normal consistency）
- densification：在一定迭代区间内进行 split/clone/prune（代码在文件后半段，策略与 GraphDECO 体系一致）

它与 `train_pbr.py` 的关系是：

- `train.py` 更偏“从稀疏点云/随机点”学习几何与外观（SH）
- `train_pbr.py` 更偏“给定几何（dense 点云），只解释材质与光照（PBR）”

---

<a id="sec-10-artifacts"></a>
## 10. 训练/渲染的产物（Artifacts）

### 10.1 训练输出目录（`--model_path`）

训练脚本都会在 `model_path` 写：

- `cfg_args`：训练参数的 `Namespace` 字符串（供 `get_combined_args` 读回）
- `input.ply`：数据集初始点云备份（从 `scene_info.ply_path` copy）
- `cameras.json`：相机列表的 JSON（train+test）
- `point_cloud/iteration_X/point_cloud.ply`：每次保存的 gaussian 点云（包含 PBR 扩展字段时也会写）

PBR 训练还会额外写：

- `env_light_{iteration}.pth`：环境贴图参数（state_dict）

### 10.2 渲染输出目录

`render_pbr.py` 会在：

- `model_path/train/ours_ITER/` 与 `model_path/test/ours_ITER/`

下创建并写入：

- `renders/`：SH 合成图（SH render + skybox）
- `pbr_shaded/`：PBR 合成图（PBR shaded + skybox）
- `gt/`：GT 图像
- `albedo/roughness/metallic/normal/depth/`：材质与几何可视化

---

<a id="sec-11-deps"></a>
## 11. 依赖与可运行性（environment.yml）

本目录同时提供了 conda 版本的 `environment.yml` 与 pip 版本的 `requirements.txt`：

- `requirements.txt`：推荐用于“服务器已有可用 PyTorch/CUDA 轮子”的场景，包含两个本地 CUDA 扩展的 editable 安装（`submodules/diff-surfel-rasterization`、`submodules/simple-knn`）。
- `environment.yml`：适合希望用 conda 一键创建环境的场景，但需要注意它对 PyTorch/CUDA/Open3D 的版本组合在某些机器上可能不可解，需要按实际服务器情况调整。

`environment.yml` 定义了一个 conda env（`surfel_splatting`），关键点：

- PyTorch + CUDA（`pytorch-cuda=13.0`，`pytorch=2.9.0`，`python=3.12`）
- `open3d`、`opencv-python`、`plyfile`、`mediapy` 等
- 两个需要编译的 pip 包（本地路径）：
  - `submodules/diff-surfel-rasterization`
  - `submodules/simple-knn`

若这些 submodule 未成功编译/安装，`gaussian_renderer` 与 `GaussianModel.create_from_dense_pcd` 会直接无法运行。

---

<a id="sec-12-tests"></a>
## 12. Tests（作为“读懂接口”的活文档）

`external/2DGS-PBR/tests/` 下的测试文件更像“组件验证脚本”，重点是：

- `test_pbr_shading.py`：验证 `pbr_utils.py` 的 BRDF、EnvironmentLight、screen-space shading 的基本性质与 shape。
- `test_integration.py`：把 PBR 组件串起来做一个“无 CUDA 的语法/结构检查 + 张量级验证”。

这类测试适合用来确认你改动后的 API/shape 没破。

---

<a id="sec-13-troubleshooting"></a>
## 13. 常见坑与排查清单

### 13.0 masked 指标的 mask 通道归一化

本仓库的图像张量通常是 `[3,H,W]`，而 `gt_alpha_mask` 通常是 `[1,H,W]`。如果在计算 masked L1/PSNR 时不把 mask 扩展到 3 通道，会导致分母少一个通道因子，从而让 masked PSNR 系统性偏低（约 `10*log10(3) ≈ 4.77dB`）。

对应修复位置：

- `utils/loss_utils.py:l1_loss`：会把单通道 mask 扩展到与图像通道数一致
- `utils/image_utils.py:psnr` / `utils/image_utils.py:mse`：同样会扩展 mask 再做归一化

### 13.1 `run_single_scene.sh` 与 `train_pbr.py` 参数不匹配

当前 `run_single_scene.sh` 调用的是 `python train_pbr.py ...`，但它没有提供 `--gt_ply`，而 `train_pbr.py` 静态几何训练是强制要求 `--gt_ply` 的（缺参会报错）。

如果你想跑通这个脚本，需要补上 `--gt_ply <dense.ply>`（以及可选 `--env_map`）。

### 13.2 忘了反预乘 G-buffer / normal

症状：

- 材质贴图整体偏暗/边缘发黑
- roughness/metallic 在物体边界附近异常
- 正则项在背景区域产生 NaN/爆炸

排查：

- 检查是否使用 `attr_pm / (alpha + eps)` 得到材质属性
- normal 是否也按 alpha 反预乘并再次 normalize

### 13.3 env_map 学习被背景监督“拉偏”

如果 GT 背景不是 HDRI，而是抠图/黑底/未定义像素：

- 默认策略：若存在 `gt_alpha_mask` 且未设置 `--supervise_background`，则重建损失只在 mask 区域计算（并用 warmup 让 env_map 先看到背景）。

如果你强制监督背景（`--supervise_background`），可能导致 env_map 学到“补偿 GT 背景”的错误光照。

### 13.4 坐标系/矩阵转置导致 ray_dir 或 normal 方向错误

症状：

- env 采样方向不对（高光跑反）
- normal 可视化看起来“整体翻转/旋转”

排查：

- `Camera.world_view_transform` 是转置存储的；`compute_ray_directions_world_from_fov()` 已处理该约定
- 任何新写的 transform 计算都要确认你使用的是 w2c 还是 c2w，以及是否需要 `.transpose(0,1)`

### 13.5 eval 中出现 NaN/Inf（快速定位）

建议打开：

- `--debug_nonfinite_dump`：自动把异常 iteration 的关键张量 dump 到 `<model_path>/debug_nonfinite/`
- `--dump_env_map_on_eval`：每次 eval 额外 dump `env_map` 到 `<model_path>/debug_env_map/`，用于定位 env_map 是否先变成 NaN/Inf 或爆亮

### 13.6 为什么“棋盘格地面”不能靠 env_map 复原？（以及怎么做）

env_map 表示的是“无限远方向 → 颜色/辐射度”，无法表达有限深度几何产生的视差（例如地面/墙面纹理）。

如果你要复原地面纹理，应使用“平面/几何 + 拼贴”：

- `scripts/reconstruct_ground_plane_texture.py`
  - `--plane_mode fit`：点云包含地面点时，RANSAC 拟合主平面并拼贴
  - `--plane_mode ymin`：点云仅含物体时，用物体最低点 + up 轴推一个地面平面，并用多视角重投影一致性筛选地面像素再拼贴

---

<a id="sec-14-reading-order"></a>
## 14. 面向改代码的“推荐阅读顺序”

1. `gaussian_renderer/__init__.py`（render 包含哪些 buffer、alpha/normal/depth 的来源）
2. `utils/pbr_utils.py`（EnvironmentLight 与 screen-space shading 的输入输出约定）
3. `train_pbr.py`（如何把 G-buffer 与 env 合成，并构造 loss/优化器）
4. `scene/gaussian_model.py`（PBR 参数如何存储/激活/保存到 PLY）
5. `scene/__init__.py` + `scene/dataset_readers.py`（数据集识别与 mask 流）

---

<a id="sec-15-file-index"></a>
## 15. 关键文件索引（按“要查什么”）

- “渲染输出有哪些字段？” → `gaussian_renderer/__init__.py`
- “out_others 通道是啥？” → `submodules/diff-surfel-rasterization/cuda_rasterizer/auxiliary.h`
- “PBR shaded 怎么算？” → `utils/pbr_utils.py:screen_space_pbr_shading`、`pbr_shading_env`
- “PBR 训练怎么组合 loss？” → `train_pbr.py` + `utils/loss_utils.py:compute_pbr_losses`
- “mask 怎么来的？” → `scene/dataset_readers.py` + `utils/camera_utils.py`
- “env_light 怎么存/怎么读？” → `train_pbr.py`（保存） + `render_pbr.py`（加载）
- “env_light 初始化/先验怎么用？” → `train_env_light.py` + `train_pbr.py`（`--env_light_pth/--env_light_prior_weight`）
- “如何启用有限深度地面背景？” → `scripts/reconstruct_ground_plane_texture.py`（生成 `ground_plane.json`/`ground_texture.png`） + `train_pbr.py`/`render_pbr.py`（`--ground_plane_json/--ground_texture`）
- “eval NaN/Inf 怎么定位？” → `train_pbr.py`（`--debug_nonfinite_dump/--dump_env_map_on_eval`） + `<model_path>/debug_nonfinite/`/`debug_env_map/`
- “棋盘格地面怎么复原？” → `scripts/reconstruct_ground_plane_texture.py`（plane mosaic）
