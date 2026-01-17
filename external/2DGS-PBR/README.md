# 2DGS-PBR (Static Geometry): Materials + Environment Light

## English

This directory contains a static-geometry PBR training/rendering pipeline:

- Initialize geometry from a dense point cloud (`--gt_ply`) and **lock** `xyz/rotation` (no densification).
- Optimize: `opacity/scale`, PBR materials (`albedo/roughness/metallic`), and a learnable environment map (`env_light`).
- Supervise the composite image: `PBR(object) * alpha + skybox(env_map) * (1-alpha)`.

### Install (pip venv)

```bash
python -m venv om
source om/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Inputs

**Dataset (`-s`)**

Supported dataset layouts:

- COLMAP: dataset root contains `sparse/`
- Blender/NeRF-synthetic: dataset root contains `transforms_train.json`

**Foreground mask (optional)**

Mask is stored as `Camera.gt_alpha_mask`:

- If images are RGBA/LA, the alpha channel is used.
- For Blender-style datasets, `mask/xxx.png` is supported (preferred over image alpha) for `images/xxx.png`.

Mask convention: foreground/object = `1`.

**Dense geometry (`--gt_ply`)**

`--gt_ply` is a dense point cloud (normals recommended):

- required: `x,y,z`
- recommended: `nx,ny,nz`
- optional: `red,green,blue` (used to initialize albedo)

If you start from a mesh, sample it into a dense PLY:

```bash
python scripts/sample_mesh_to_ply.py \
  --input_mesh <path_to_mesh.(obj|glb|ply|stl)> \
  --output_ply <path_to_dense_point_cloud.ply> \
  --num_points 1000000
```

### Train (static-geometry PBR)

Example (replace placeholders):

```bash
python train_pbr.py \
  -s <DATASET_ROOT>/<hash>/<scene_name> \
  -m output/<scene_name>_pbr_geo_static/1/ \
  --gt_ply <GT_ROOT>/<hash>/dense_sampled.ply \
  --eval \
  --test_interval 1000 \
  --scaling_lr 0.001 \
  --roughness_lr 0.01 \
  --supervise_background
```

#### `train_pbr.py` argument reference

**Core paths**

- `-s/--source_path`: dataset path
- `-m/--model_path`: output path
- `--gt_ply`: dense point cloud path

**Run control**

- `--iterations`: total iterations
- `--save_iterations <i1 i2 ...>`: save checkpoints at these iterations
- `--test_iterations <i1 i2 ...>`: run evaluation at these iterations
- `--test_interval <N>`: run evaluation every `N` iterations (overrides `--test_iterations`)
- `--eval`: enable train/test split for supported datasets
- `--eval_first`: run evaluation once at iteration `0` (before any optimizer steps)

**Learning rates (gaussians / materials / env)**

- `--opacity_lr`: opacity LR
- `--scaling_lr`: scaling LR
- `--albedo_lr`: albedo LR
- `--roughness_lr`: roughness LR
- `--metallic_lr`: metallic LR
- `--env_light_lr`: environment map LR

**Initialization**

- `--roughness_init`: initial roughness value in `[0,1]` (applied to all points before training)

**Environment light**

- `--env_map`: initial environment map (`.hdr/.exr/.png`); if omitted, starts from a learnable gray map
- `--env_map_res`: env_map resolution height (width is `2*H`)
- `--lambda_env_tv`: solid-angle weighted TV regularization weight
- `--lambda_env_smooth`: solid-angle weighted Laplacian regularization weight
- `--env_clamp_min`, `--env_clamp_max`: clamp env_map values after each step
- `--no_env_gradient_scaling`: disable solid-angle gradient scaling hook
- `--env_warmup_iters`: warmup iterations where only env_map is optimized

**Reconstruction supervision (composite image)**

- `--lambda_rgb`: weight for composite reconstruction loss
- `--lambda_dssim`: SSIM mixing weight (reconstruction is `(1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM)`)
- `--supervise_background`: supervise full composite even when `gt_alpha_mask` exists
- `--lambda_pbr`: extra reconstruction weight on object region (mask if available, else alpha)
- `--lambda_alpha`: supervise `rend_alpha` toward `gt_alpha_mask`

**Material regularization**

- `--lambda_pbr_reg`: global scale on material regularization
- `--lambda_albedo_smooth`
- `--lambda_roughness_smooth`
- `--lambda_metallic_smooth`
- `--lambda_metallic_prior`
- `--lambda_roughness_prior`
- `--lambda_albedo_chroma`

**Parameter bounds / stability**

- `--roughness_min`, `--roughness_max`: clamp range for roughness
- `--lambda_scale_reg`: soft penalty for oversized gaussians
- `--scale_reg_max_ratio`: scale threshold ratio (relative to scene extent) used by `lambda_scale_reg`
- `--scale_clamp_max_ratio`: hard clamp on gaussian scales (ratio * scene extent)

**Early stopping**

- `--enable_early_stopping`
- `--early_stopping_patience`
- `--early_stopping_min_delta`
- `--early_stopping_interval`

**Logging / debug**

- `--log_interval`: print detailed breakdown every N iterations (`0` disables)
- `--quiet`: reduce console output
- `--debug_nonfinite_dump`: write a debug dump when NaN/Inf appears during evaluation
- `--debug_nonfinite_dump_full`: include full-resolution tensors in the dump (can be very large)
- `--debug_nonfinite_raise`: raise after writing the dump
- `--dump_env_map_on_eval`: save `env_light.env_map` to `debug_env_map/` at each evaluation iteration

#### Loss function reference (what is optimized)

`train_pbr.py` optimizes a weighted sum of reconstruction + regularization terms:

**1) Composite reconstruction loss**

- `pred = shaded_obj * alpha + bg_env * (1 - alpha)`
- `recon_loss = (1 - lambda_dssim) * L1(pred, gt; weight=recon_weight) + lambda_dssim * (1 - SSIM(pred, gt; weight=recon_weight))`
- `recon_weight`:
  - warmup (`iter <= env_warmup_iters`): full image (`ones`)
  - otherwise: if `gt_alpha_mask` exists and `--supervise_background` is NOT set, use GT mask (foreground-only)
  - otherwise: full image; optionally increased on object region by `--lambda_pbr`

**2) Alpha supervision (optional)**

- `alpha_sup_loss = lambda_alpha * mean(|rend_alpha - gt_alpha_mask|)` (when GT mask exists and enabled)

**3) Environment map regularization**

- `env_tv_loss = lambda_env_tv * tv_loss_weighted(env_map)`
- `env_smooth_loss = lambda_env_smooth * smoothness_loss_weighted(env_map)`

**4) Material regularization (masked)**

- `reg_mask = gt_alpha_mask if available else rend_alpha.detach()`
- `compute_pbr_losses(...)` returns: `albedo_smooth`, `roughness_smooth`, `metallic_smooth`, `metallic_prior`, `roughness_prior`, `albedo_chroma`, `total_pbr_reg`
- `pbr_reg_loss = lambda_pbr_reg * total_pbr_reg`

**5) Scale blow-up prevention (optional)**

- threshold is `scale_reg_max_ratio * scene_extent`
- `scale_reg_loss = lambda_scale_reg * mean(relu(log_scale_max - log_thresh)^2)`

**Total loss**

`total_loss = lambda_rgb * recon_loss + env_tv_loss + env_smooth_loss + scale_reg_loss + alpha_sup_loss + pbr_reg_loss`

Outputs (under `-m`):

- `point_cloud/iteration_<iter>/point_cloud.ply`
- `env_light_<iter>.pth`

### Render (PBR outputs)

```bash
python render_pbr.py -m <output_path> --compute_metrics
```

#### `render_pbr.py` argument reference

- `-m/--model_path`: output path to render from
- `--iteration`: which iteration to render (`-1` selects latest)
- `--env_map`: override env map path (used when no saved `env_light_*.pth` is found)
- `--compute_metrics`: compute PSNR/SSIM (and LPIPS if available)
- `--skip_train`, `--skip_test`: skip rendering those splits

Outputs are written under:

- `<output_path>/train/ours_<iter>/...`
- `<output_path>/test/ours_<iter>/...`

including `pbr_shaded/`, `albedo/`, `roughness/`, `metallic/`, `normal/`, `depth/`.

---

## 中文

本目录包含一个“静态几何”的 PBR 训练/渲染流程：

- 从 dense 点云 `--gt_ply` 初始化几何，并在训练中 **锁定** `xyz/rotation`（不做 densification）。
- 只优化：`opacity/scale`、PBR 材质（`albedo/roughness/metallic`）与可学习环境贴图（`env_light`）。
- 监督的是合成图：`PBR(object) * alpha + skybox(env_map) * (1-alpha)`。

### 安装（pip venv）

```bash
python -m venv om
source om/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 输入

**数据集（`-s`）**

支持的数据目录结构：

- COLMAP：数据集根目录包含 `sparse/`
- Blender/NeRF Synthetic：数据集根目录包含 `transforms_train.json`

**前景 mask（可选）**

mask 会存到 `Camera.gt_alpha_mask`：

- 如果图片是 RGBA/LA，使用 alpha 通道作为 mask。
- 对 Blender 风格数据，支持 `images/xxx.png` 对应 `mask/xxx.png`（优先级高于图片 alpha）。

mask 约定：前景/物体 = `1`。

**稠密几何（`--gt_ply`）**

`--gt_ply` 是稠密点云（强烈建议带法线）：

- 必需：`x,y,z`
- 建议：`nx,ny,nz`
- 可选：`red,green,blue`（用于初始化 albedo）

如果你的输入是 mesh，可采样成 dense PLY：

```bash
python scripts/sample_mesh_to_ply.py \
  --input_mesh <path_to_mesh.(obj|glb|ply|stl)> \
  --output_ply <path_to_dense_point_cloud.ply> \
  --num_points 1000000
```

### 训练（静态几何 PBR）

运行示例（替换占位符）：

```bash
python train_pbr.py \
  -s <DATASET_ROOT>/<hash>/<scene_name> \
  -m output/<scene_name>_pbr_geo_static/1/ \
  --gt_ply <GT_ROOT>/<hash>/dense_sampled.ply \
  --eval \
  --test_interval 1000 \
  --scaling_lr 0.001 \
  --roughness_lr 0.01 \
  --supervise_background
```

#### `train_pbr.py` 参数说明

**核心路径参数**

- `-s/--source_path`：数据集路径
- `-m/--model_path`：输出目录
- `--gt_ply`：稠密点云路径

**运行控制**

- `--iterations`：总迭代步数
- `--save_iterations <i1 i2 ...>`：在指定步保存
- `--test_iterations <i1 i2 ...>`：在指定步评测
- `--test_interval <N>`：每 `N` 步评测一次（覆盖 `--test_iterations`）
- `--eval`：启用 train/test 切分（依赖数据集 loader）

**学习率（gaussians/材质/env）**

- `--opacity_lr`：opacity 学习率
- `--scaling_lr`：scaling 学习率
- `--albedo_lr`：albedo 学习率
- `--roughness_lr`：roughness 学习率
- `--metallic_lr`：metallic 学习率
- `--env_light_lr`：环境贴图学习率

**环境贴图（env light）**

- `--env_map`：环境贴图初始化文件（`.hdr/.exr/.png`）；不提供则从可学习的灰色 env 开始
- `--env_map_res`：环境贴图高度分辨率（宽度为 `2*H`）
- `--lambda_env_tv`：TV 正则权重（solid-angle 加权，适配经纬投影）
- `--lambda_env_smooth`：Laplacian 平滑正则权重（solid-angle 加权）
- `--env_clamp_min`, `--env_clamp_max`：每步更新后对 env_map 值做 clamp
- `--no_env_gradient_scaling`：关闭 solid-angle 梯度缩放 hook
- `--env_warmup_iters`：warmup 步数（warmup 期只优化 env_map）

**重建监督（合成图）**

- `--lambda_rgb`：合成图重建损失权重
- `--lambda_dssim`：SSIM 混合系数（重建为 `(1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM)`）
- `--supervise_background`：即使有 `gt_alpha_mask` 也监督全图
- `--lambda_pbr`：对物体区域额外加权（优先 GT mask，否则用 alpha）
- `--lambda_alpha`：alpha 监督：让 `rend_alpha` 逼近 `gt_alpha_mask`

**材质正则**

- `--lambda_pbr_reg`：材质正则的总缩放权重
- `--lambda_albedo_smooth`
- `--lambda_roughness_smooth`
- `--lambda_metallic_smooth`
- `--lambda_metallic_prior`
- `--lambda_roughness_prior`
- `--lambda_albedo_chroma`

**参数范围与稳定性**

- `--roughness_min`, `--roughness_max`：roughness clamp 范围
- `--lambda_scale_reg`：防止 gaussian 过大的 soft penalty
- `--scale_reg_max_ratio`：尺度阈值比例（相对 scene extent），配合 `lambda_scale_reg`
- `--scale_clamp_max_ratio`：对 scale 做 hard clamp（比例 * scene extent）

**早停**

- `--enable_early_stopping`
- `--early_stopping_patience`
- `--early_stopping_min_delta`
- `--early_stopping_interval`

**日志与调试**

- `--log_interval`：每 N 步打印一次详细 loss 分解（`0` 关闭）
- `--quiet`：减少控制台输出

#### 损失函数说明（训练到底在优化什么）

`train_pbr.py` 的总损失是“重建项 + 正则项”的加权和：

**1）合成图重建损失**

- `pred = shaded_obj * alpha + bg_env * (1 - alpha)`
- `recon_loss = (1 - lambda_dssim) * L1(pred, gt; weight=recon_weight) + lambda_dssim * (1 - SSIM(pred, gt; weight=recon_weight))`
- `recon_weight`：
  - warmup（`iter <= env_warmup_iters`）：全图（全 1）
  - 否则：若存在 `gt_alpha_mask` 且未开 `--supervise_background`，仅在 GT mask（前景）区域监督
  - 否则：全图；并可用 `--lambda_pbr` 对物体区域额外加权

**2）Alpha 监督（可选）**

- `alpha_sup_loss = lambda_alpha * mean(|rend_alpha - gt_alpha_mask|)`（存在 GT mask 且启用时）

**3）环境贴图正则**

- `env_tv_loss = lambda_env_tv * tv_loss_weighted(env_map)`
- `env_smooth_loss = lambda_env_smooth * smoothness_loss_weighted(env_map)`

**4）材质正则（带 mask）**

- `reg_mask = gt_alpha_mask (若存在)；否则使用 rend_alpha.detach()`
- `compute_pbr_losses(...)` 分项：`albedo_smooth`, `roughness_smooth`, `metallic_smooth`, `metallic_prior`, `roughness_prior`, `albedo_chroma`, `total_pbr_reg`
- `pbr_reg_loss = lambda_pbr_reg * total_pbr_reg`

**5）scale 爆炸抑制（可选）**

- 阈值为 `scale_reg_max_ratio * scene_extent`
- `scale_reg_loss = lambda_scale_reg * mean(relu(log_scale_max - log_thresh)^2)`

**总损失**

`total_loss = lambda_rgb * recon_loss + env_tv_loss + env_smooth_loss + scale_reg_loss + alpha_sup_loss + pbr_reg_loss`

输出（在 `-m` 目录下）：

- `point_cloud/iteration_<iter>/point_cloud.ply`
- `env_light_<iter>.pth`

### 渲染（PBR 输出）

```bash
python render_pbr.py -m <output_path> --compute_metrics
```

#### `render_pbr.py` 参数说明

- `-m/--model_path`：渲染输入目录
- `--iteration`：指定迭代（`-1` 自动选择最新）
- `--env_map`：找不到保存的 `env_light_*.pth` 时用外部 env_map
- `--compute_metrics`：计算 PSNR/SSIM（可选 LPIPS）
- `--skip_train`, `--skip_test`：跳过对应 split

输出目录：

- `<output_path>/train/ours_<iter>/...`
- `<output_path>/test/ours_<iter>/...`

包含：`pbr_shaded/`, `albedo/`, `roughness/`, `metallic/`, `normal/`, `depth/`。
