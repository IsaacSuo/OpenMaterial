# 2DGS-PBR Pipeline Quick Reference

本页是 `docs/PIPELINE_REFERENCE.md` 的索引入口：按“我现在要做什么/要查什么”快速跳到对应章节。

## Start Here

- 全局心智模型（先读这个）→ [PIPELINE_REFERENCE.md#sec-0-mental-model](PIPELINE_REFERENCE.md#sec-0-mental-model)
- 目录与模块职责（Repo Map）→ [PIPELINE_REFERENCE.md#sec-1-repo-map](PIPELINE_REFERENCE.md#sec-1-repo-map)
- 张量/形状/坐标约定（最常踩坑）→ [PIPELINE_REFERENCE.md#sec-2-conventions](PIPELINE_REFERENCE.md#sec-2-conventions)

## Training

- PBR 静态几何训练（`train_pbr.py` 全流程）→ [PIPELINE_REFERENCE.md#sec-7-train-pbr](PIPELINE_REFERENCE.md#sec-7-train-pbr)
- 环境光预训练/初始化（`train_env_light.py`）→ [PIPELINE_REFERENCE.md#sec-7-train-pbr](PIPELINE_REFERENCE.md#sec-7-train-pbr)
- 有限深度地面背景（`GroundPlane` / `--ground_plane_json`）→ [PIPELINE_REFERENCE.md#sec-7-train-pbr](PIPELINE_REFERENCE.md#sec-7-train-pbr)
- 可学习的有限深度背景（Unfixed Background Gaussians，SH-only：`--unfixed_gaussians`）→ [PIPELINE_REFERENCE.md#sec-7-train-pbr](PIPELINE_REFERENCE.md#sec-7-train-pbr)
- 基础 2DGS 训练（`train.py`，与 PBR 的关系）→ [PIPELINE_REFERENCE.md#sec-9-train-2dgs](PIPELINE_REFERENCE.md#sec-9-train-2dgs)
- 训练/渲染产物（输出目录里有什么）→ [PIPELINE_REFERENCE.md#sec-10-artifacts](PIPELINE_REFERENCE.md#sec-10-artifacts)

## Rendering

- PBR 渲染导出与指标（`render_pbr.py`）→ [PIPELINE_REFERENCE.md#sec-8-render-pbr](PIPELINE_REFERENCE.md#sec-8-render-pbr)
- 有限深度地面背景渲染（`--ground_plane_json`）→ [PIPELINE_REFERENCE.md#sec-8-render-pbr](PIPELINE_REFERENCE.md#sec-8-render-pbr)
- 光栅化输出包字段（alpha/normal/depth/distortion/G-buffer）→ [PIPELINE_REFERENCE.md#sec-2-conventions](PIPELINE_REFERENCE.md#sec-2-conventions)

## PBR & Materials

- PBR 着色（G-buffer → shaded → 与 skybox 合成）→ [PIPELINE_REFERENCE.md#sec-6-pbr-shading](PIPELINE_REFERENCE.md#sec-6-pbr-shading)
- `GaussianModel` 的 PBR 参数（存储/激活/optimizer/PLY 字段）→ [PIPELINE_REFERENCE.md#sec-5-gaussian-model](PIPELINE_REFERENCE.md#sec-5-gaussian-model)

## Data & Formats

- COLMAP / Blender 数据读取与 mask 流 → [PIPELINE_REFERENCE.md#sec-3-data-formats](PIPELINE_REFERENCE.md#sec-3-data-formats)
- 坐标系与 transform 转置约定（ray_dir/normal 方向问题）→ [PIPELINE_REFERENCE.md#sec-4-coordinates](PIPELINE_REFERENCE.md#sec-4-coordinates)

## Debugging

- 常见坑与排查清单（含 `run_single_scene.sh` 缺 `--gt_ply`）→ [PIPELINE_REFERENCE.md#sec-13-troubleshooting](PIPELINE_REFERENCE.md#sec-13-troubleshooting)
- 背景/地面纹理复原（棋盘格地面）→ [PIPELINE_REFERENCE.md#sec-13-troubleshooting](PIPELINE_REFERENCE.md#sec-13-troubleshooting)
- eval NaN/Inf 自动 dump（`--debug_nonfinite_dump` / `--dump_env_map_on_eval`）→ [PIPELINE_REFERENCE.md#sec-13-troubleshooting](PIPELINE_REFERENCE.md#sec-13-troubleshooting)
- Tests（把测试当成接口契约）→ [PIPELINE_REFERENCE.md#sec-12-tests](PIPELINE_REFERENCE.md#sec-12-tests)

## Environment & Build

- 依赖/环境（`requirements.txt` / `environment.yml`、CUDA 扩展 submodules）→ [PIPELINE_REFERENCE.md#sec-11-deps](PIPELINE_REFERENCE.md#sec-11-deps)

## Lookup Tables

- 关键文件索引（按“要查什么”）→ [PIPELINE_REFERENCE.md#sec-15-file-index](PIPELINE_REFERENCE.md#sec-15-file-index)
- 推荐阅读顺序（适合第一次通读代码）→ [PIPELINE_REFERENCE.md#sec-14-reading-order](PIPELINE_REFERENCE.md#sec-14-reading-order)
