#!/usr/bin/env python3
"""
train_pbr_v2.py

Config-driven entrypoint for static-geometry 2DGS-PBR training.

Goals:
  - Keep the training implementation in train_pbr.py (training_pbr_static) as the engine
  - Replace the CLI "flag soup" with a JSON config + small CLI override surface
  - Make runs reproducible: save resolved config next to outputs

Usage:
  python train_pbr_v2.py --config configs/pbr_env_probe_unfixed_complex.json

Override examples:
  python train_pbr_v2.py --config cfg.json --override optim.iters=60000
  python train_pbr_v2.py --config cfg.json --override schedule.eval.every=1000 --override lighting.object.probe.lr=0.005
  python train_pbr_v2.py --config cfg.json --override background.unfixed.lambda_obj_overlap=0.05

Notes:
  - Values in --override are parsed as JSON if possible (numbers, booleans, null, lists, dicts).
    If JSON parsing fails, the raw string is used.
  - This script does not try to install dependencies. If tinycudann is missing and you configured
    probe backend "tcnn", it will fallback to a pure PyTorch MLP probe (see utils/light_probe.py).
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from typing import Any

from arguments import ModelParams, OptimizationParams, PipelineParams
from train_pbr import training_pbr_static


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object/dict, got: {type(data)}")
    return data


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _set_by_dotted_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    if not dotted or dotted.strip() == "":
        raise ValueError("Override key is empty")
    parts = dotted.split(".")
    cur: Any = obj
    for i, p in enumerate(parts):
        if p.strip() == "":
            raise ValueError(f"Invalid dotted key: {dotted}")
        is_last = i == len(parts) - 1
        if is_last:
            if not isinstance(cur, dict):
                raise ValueError(f"Override path points into non-dict at '{'.'.join(parts[:i])}'")
            cur[p] = value
            return
        if not isinstance(cur, dict):
            raise ValueError(f"Override path points into non-dict at '{'.'.join(parts[:i])}'")
        if p not in cur or cur[p] is None:
            cur[p] = {}
        if not isinstance(cur[p], dict):
            raise ValueError(f"Override path conflicts at '{'.'.join(parts[:i+1])}': expected dict")
        cur = cur[p]


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _require_str(cfg: dict[str, Any], path: str) -> str:
    v = _get(cfg, path, None)
    if not isinstance(v, str) or v.strip() == "":
        raise ValueError(f"Missing/invalid required config field: {path}")
    return v


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    raise ValueError(f"Expected boolean, got {type(v)}: {v}")


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, (int, float)) and int(v) == v:
        return int(v)
    if isinstance(v, int):
        return int(v)
    raise ValueError(f"Expected int, got {type(v)}: {v}")


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    raise ValueError(f"Expected float, got {type(v)}: {v}")


def _as_str(v: Any, default: str) -> str:
    if v is None:
        return str(default)
    if isinstance(v, str):
        return v
    raise ValueError(f"Expected str, got {type(v)}: {v}")


def _as_list_int(v: Any, default: list[int]) -> list[int]:
    if v is None:
        return list(default)
    if not isinstance(v, list):
        raise ValueError(f"Expected list, got {type(v)}: {v}")
    out: list[int] = []
    for x in v:
        if isinstance(x, (int, float)) and int(x) == x:
            out.append(int(x))
        else:
            raise ValueError(f"Expected list of ints, got element {type(x)}: {x}")
    return out


def _phase_iters(cfg: dict[str, Any], phase_name: str) -> int:
    phases = _get(cfg, "schedule.phases", [])
    if not isinstance(phases, list):
        return 0
    total = 0
    for ph in phases:
        if not isinstance(ph, dict):
            continue
        if str(ph.get("name", "")).strip() == phase_name:
            total += _as_int(ph.get("iters", 0), 0)
    return int(total)


def _build_args_from_config(cfg: dict[str, Any]) -> Namespace:
    # ---- Required paths ----
    source_path = _require_str(cfg, "paths.source")
    model_path = _require_str(cfg, "paths.output")
    gt_ply = _require_str(cfg, "paths.gt_ply")
    init_env_map = _get(cfg, "paths.init_env_map", None)
    if init_env_map is not None and not isinstance(init_env_map, str):
        raise ValueError("paths.init_env_map must be a string or null")

    # ---- High-level switches ----
    object_mode = _as_str(_get(cfg, "object.mode", "pbr"), "pbr").lower().strip()
    if object_mode not in ("pbr", "sh"):
        raise ValueError("object.mode must be 'pbr' or 'sh'")

    light_object_model = _as_str(_get(cfg, "lighting.object.model", "envmap"), "envmap").lower().strip()
    if light_object_model not in ("envmap", "probe"):
        raise ValueError("lighting.object.model must be 'envmap' or 'probe'")
    if light_object_model == "probe" and object_mode != "pbr":
        raise ValueError("lighting.object.model='probe' requires object.mode='pbr'")

    # ---- Iterations / schedule ----
    stage0_bg_iters = _as_int(_get(cfg, "schedule.stage0_bg_iters", None), _phase_iters(cfg, "bg_only"))
    stage1_env_iters = _as_int(_get(cfg, "schedule.stage1_env_iters", None), _phase_iters(cfg, "env_only"))
    total_from_phases = stage0_bg_iters + stage1_env_iters + _phase_iters(cfg, "full")
    iters = _as_int(_get(cfg, "optim.iters", None), total_from_phases if total_from_phases > 0 else 30_000)

    # ---- ModelParams defaults ----
    # Keep consistent with arguments/ModelParams defaults unless overridden.
    sh_degree = _as_int(_get(cfg, "model.sh_degree", None), 3)
    images = _as_str(_get(cfg, "model.images", None), "images")
    resolution = _as_int(_get(cfg, "model.resolution", None), -1)
    white_background = _as_bool(_get(cfg, "model.white_background", None), False)
    data_device = _as_str(_get(cfg, "model.data_device", None), "cuda")
    do_eval = _as_bool(_get(cfg, "model.eval", None), False)

    # ---- PipelineParams ----
    depth_ratio = _as_float(_get(cfg, "pipeline.depth_ratio", None), 0.0)
    compute_cov3d_python = _as_bool(_get(cfg, "pipeline.compute_cov3d_python", None), False)

    # ---- OptimParams base (the engine extracts many defaults; we set the ones we care about) ----
    batch_cams = _as_int(_get(cfg, "optim.batch_cams", None), 1)
    scaling_lr = _as_float(_get(cfg, "optim.lr.scaling", None), 0.005)
    opacity_lr = _as_float(_get(cfg, "optim.lr.opacity", None), 0.05)
    albedo_lr = _as_float(_get(cfg, "optim.lr.albedo", None), 0.001)
    roughness_lr = _as_float(_get(cfg, "optim.lr.roughness", None), 0.0002)
    metallic_lr = _as_float(_get(cfg, "optim.lr.metallic", None), 0.0002)

    # ---- Loss / regularization ----
    lambda_rgb = _as_float(_get(cfg, "loss.lambda_rgb", None), 1.0)
    lambda_pbr = _as_float(_get(cfg, "loss.lambda_pbr_boost", None), 0.1)
    lambda_pbr_reg = _as_float(_get(cfg, "loss.pbr_reg.lambda_total", None), 0.01)
    lambda_bg = _as_float(_get(cfg, "loss.lambda_bg", None), 1.0)
    composite_use_gt_mask = _as_bool(_get(cfg, "loss.composite_use_gt_mask", None), False)
    supervise_background = _as_bool(_get(cfg, "loss.supervise_background", None), False)
    lambda_alpha = _as_float(_get(cfg, "loss.lambda_alpha", None), 0.0)

    pbr_reg = _get(cfg, "loss.pbr_reg", {}) or {}
    if not isinstance(pbr_reg, dict):
        raise ValueError("loss.pbr_reg must be an object/dict")
    lambda_albedo_smooth = _as_float(pbr_reg.get("albedo_smooth", None), 0.01)
    lambda_roughness_smooth = _as_float(pbr_reg.get("roughness_smooth", None), 0.01)
    lambda_metallic_smooth = _as_float(pbr_reg.get("metallic_smooth", None), 0.01)
    lambda_metallic_prior = _as_float(pbr_reg.get("metallic_prior", None), 0.001)
    lambda_roughness_prior = _as_float(pbr_reg.get("roughness_prior", None), 0.001)
    lambda_albedo_chroma = _as_float(pbr_reg.get("albedo_chroma", None), 0.001)

    # ---- EnvMap (sky lighting) ----
    env_cfg = _get(cfg, "lighting.sky.envmap", {}) or {}
    if not isinstance(env_cfg, dict):
        raise ValueError("lighting.sky.envmap must be an object/dict")
    env_map_res = _as_int(env_cfg.get("res_h", None), 1024)
    env_light_lr = _as_float(env_cfg.get("lr", None), 0.01)
    lambda_env_tv = _as_float(env_cfg.get("lambda_tv", None), 0.001)
    lambda_env_smooth = _as_float(env_cfg.get("lambda_smooth", None), 0.0)
    env_clamp_min = env_cfg.get("clamp_min", None)
    env_clamp_max = env_cfg.get("clamp_max", None)
    freeze_env_light = _as_bool(env_cfg.get("freeze", None), False)
    env_light_pth = env_cfg.get("prior_pth", None)
    if env_light_pth is not None and not isinstance(env_light_pth, str):
        raise ValueError("lighting.sky.envmap.prior_pth must be a string or null")
    env_light_prior_weight = _as_float(env_cfg.get("prior_weight", None), 0.0)
    env_light_prior_log_space = _as_bool(env_cfg.get("prior_log_space", None), False)

    env_update_after = _as_int(_get(cfg, "schedule.env_update.after", None), 0)
    env_update_interval = _as_int(_get(cfg, "schedule.env_update.interval", None), 1)

    # ---- Probe (object lighting) ----
    probe_cfg = _get(cfg, "lighting.object.probe", {}) or {}
    if not isinstance(probe_cfg, dict):
        raise ValueError("lighting.object.probe must be an object/dict")
    probe_backend = _as_str(_get(cfg, "lighting.object.probe.backend", None), "tcnn")
    probe_lr = _as_float(_get(cfg, "lighting.object.probe.lr", None), 1e-2)
    probe_weight_decay = _as_float(_get(cfg, "lighting.object.probe.weight_decay", None), 0.0)
    probe_update_after = _as_int(_get(cfg, "lighting.object.probe.update_after", None), 0)
    probe_update_interval = _as_int(_get(cfg, "lighting.object.probe.update_interval", None), 1)
    freeze_probe = _as_bool(_get(cfg, "lighting.object.probe.freeze", None), False)
    probe_pth = _get(cfg, "lighting.object.probe.ckpt", None)
    if probe_pth is not None and not isinstance(probe_pth, str):
        raise ValueError("lighting.object.probe.ckpt must be a string or null")

    probe_tcnn = _get(cfg, "lighting.object.probe.tcnn", {}) or {}
    if not isinstance(probe_tcnn, dict):
        raise ValueError("lighting.object.probe.tcnn must be an object/dict")
    probe_dir_encoding = _as_str(probe_tcnn.get("dir_encoding", None), "sh")
    probe_sh_degree = _as_int(probe_tcnn.get("sh_degree", None), 4)
    probe_fourier_n_freq = _as_int(probe_tcnn.get("fourier_n_frequencies", None), 6)
    probe_n_levels = _as_int(probe_tcnn.get("n_levels", None), 16)
    probe_n_feat = _as_int(probe_tcnn.get("n_features_per_level", None), 2)
    probe_log2_hash = _as_int(probe_tcnn.get("log2_hashmap_size", None), 19)
    probe_base_res = _as_int(probe_tcnn.get("base_resolution", None), 16)
    probe_per_level_scale = _as_float(probe_tcnn.get("per_level_scale", None), 1.5)

    probe_mlp = _get(cfg, "lighting.object.probe.mlp", {}) or {}
    if not isinstance(probe_mlp, dict):
        raise ValueError("lighting.object.probe.mlp must be an object/dict")
    probe_hidden_dim = _as_int(probe_mlp.get("hidden_dim", None), 64)
    probe_n_hidden = _as_int(probe_mlp.get("n_hidden_layers", None), 2)

    probe_out = _get(cfg, "lighting.object.probe.output", {}) or {}
    if not isinstance(probe_out, dict):
        raise ValueError("lighting.object.probe.output must be an object/dict")
    probe_out_act = _as_str(probe_out.get("activation", None), "softplus")
    probe_out_beta = _as_float(probe_out.get("softplus_beta", None), 1.0)

    probe_aabb = _get(cfg, "lighting.object.probe.aabb", {}) or {}
    if not isinstance(probe_aabb, dict):
        raise ValueError("lighting.object.probe.aabb must be an object/dict")
    probe_aabb_margin_ratio = _as_float(probe_aabb.get("margin_ratio", None), 0.05)
    probe_aabb_min = probe_aabb.get("min", None)
    probe_aabb_max = probe_aabb.get("max", None)

    # ---- Background: unfixed gaussians ----
    bg_model = _as_str(_get(cfg, "background.model", None), "none").lower().strip()
    unfixed_gaussians = bg_model == "unfixed"
    unfixed_cfg = _get(cfg, "background.unfixed", {}) or {}
    if unfixed_cfg is not None and not isinstance(unfixed_cfg, dict):
        raise ValueError("background.unfixed must be an object/dict")
    unfixed_num_points = _as_int(unfixed_cfg.get("num_points", None), 200_000)
    unfixed_seed = _as_int(unfixed_cfg.get("seed", None), 0)
    unfixed_exclude_ratio = _as_float(unfixed_cfg.get("exclude_object_aabb_margin_ratio", None), 0.02)
    unfixed_disable_densification = _as_bool(unfixed_cfg.get("disable_densification", None), False)
    unfixed_disable_opacity_reset = _as_bool(unfixed_cfg.get("disable_opacity_reset", None), False)
    lambda_unfixed_obj_overlap = _as_float(unfixed_cfg.get("lambda_obj_overlap", None), 0.0)

    # ---- Debug / eval / save ----
    save_at = _as_list_int(_get(cfg, "schedule.save.at", None), [7_000, iters])
    test_every = _as_int(_get(cfg, "schedule.eval.every", None), 0)
    test_at = _as_list_int(_get(cfg, "schedule.eval.at", None), [7_000, iters])
    eval_first = _as_bool(_get(cfg, "schedule.eval.first", None), False)

    quiet = _as_bool(_get(cfg, "debug.quiet", None), False)
    log_interval = _as_int(_get(cfg, "debug.log_every", None), 500)
    debug_nonfinite_dump = _as_bool(_get(cfg, "debug.dump_nonfinite", None), False)
    debug_nonfinite_dump_full = _as_bool(_get(cfg, "debug.dump_nonfinite_full", None), False)
    debug_nonfinite_raise = _as_bool(_get(cfg, "debug.dump_nonfinite_raise", None), False)
    dump_env_map_on_eval = _as_bool(_get(cfg, "debug.dump_env_map_on_eval", None), False)
    tb_log_env_map = _as_bool(_get(cfg, "debug.tb_log_env_map", None), False)

    # ---- Early stopping (optional) ----
    es = _get(cfg, "schedule.early_stopping", {}) or {}
    if not isinstance(es, dict):
        raise ValueError("schedule.early_stopping must be an object/dict")
    enable_early_stopping = _as_bool(es.get("enable", None), False)
    early_stopping_patience = _as_int(es.get("patience", None), 10)
    early_stopping_min_delta = _as_float(es.get("min_delta", None), 0.01)
    early_stopping_interval = _as_int(es.get("interval", None), 500)

    # ---- Assemble args namespace ----
    args = Namespace()
    # ModelParams
    args.source_path = os.path.abspath(source_path)
    args.model_path = model_path
    args.images = images
    args.resolution = resolution
    args.white_background = white_background
    args.sh_degree = sh_degree
    args.data_device = data_device
    args.eval = do_eval

    # OptimizationParams (core)
    args.iterations = iters
    args.opacity_lr = opacity_lr
    args.scaling_lr = scaling_lr
    args.percent_dense = 0.01  # default; keep opt extractor happy
    args.lambda_dssim = _as_float(_get(cfg, "loss.lambda_dssim", None), 0.2)
    args.lambda_dist = 0.0
    args.lambda_normal = 0.05
    args.opacity_cull = 0.05
    args.densification_interval = 100
    args.opacity_reset_interval = 3000
    args.densify_from_iter = 500
    args.densify_until_iter = 15_000
    args.densify_grad_threshold = 0.0002

    # PipelineParams
    args.convert_SHs_python = False
    args.compute_cov3D_python = compute_cov3d_python
    args.depth_ratio = depth_ratio
    args.debug = False

    # Static PBR engine extra args (directly referenced in train_pbr.py)
    args.gt_ply = gt_ply
    args.env_map = init_env_map
    args.env_map_res = env_map_res
    args.env_light_pth = env_light_pth
    args.no_env_gradient_scaling = False
    args.early_stopping_interval = early_stopping_interval
    args.lambda_bg = lambda_bg
    args.lambda_albedo_smooth = lambda_albedo_smooth
    args.lambda_roughness_smooth = lambda_roughness_smooth
    args.lambda_metallic_smooth = lambda_metallic_smooth
    args.lambda_metallic_prior = lambda_metallic_prior
    args.lambda_roughness_prior = lambda_roughness_prior
    args.lambda_albedo_chroma = lambda_albedo_chroma
    args.log_interval = log_interval
    args.enable_early_stopping = enable_early_stopping
    args.early_stopping_min_delta = early_stopping_min_delta
    args.early_stopping_patience = early_stopping_patience
    args.save_iterations = list(save_at)
    args.test_iterations = list(test_at)
    args.test_interval = test_every

    # PBR-specific weights that are transferred into opt in train_pbr.py main
    args.lambda_rgb = lambda_rgb
    args.env_light_lr = env_light_lr
    args.lambda_pbr = lambda_pbr
    args.lambda_pbr_reg = lambda_pbr_reg
    args.lambda_env_tv = lambda_env_tv
    args.albedo_lr = albedo_lr
    args.roughness_lr = roughness_lr
    args.metallic_lr = metallic_lr

    # Additional knobs used via getattr in training loop
    args.lambda_alpha = lambda_alpha
    args.lambda_env_smooth = lambda_env_smooth
    args.env_clamp_min = env_clamp_min
    args.env_clamp_max = env_clamp_max
    args.freeze_env_light = freeze_env_light
    args.env_light_prior_weight = env_light_prior_weight
    args.env_light_prior_log_space = env_light_prior_log_space
    args.env_update_after = env_update_after
    args.env_update_interval = env_update_interval
    args.batch_cams = batch_cams
    args.supervise_background = supervise_background
    args.composite_use_gt_mask = composite_use_gt_mask
    args.object_render_mode = object_mode
    args.roughness_init = _get(cfg, "object.roughness_init", None)
    args.roughness_min = _as_float(_get(cfg, "object.roughness_min", None), 0.02)
    args.roughness_max = _as_float(_get(cfg, "object.roughness_max", None), 0.999)

    # Schedule staging
    args.stage0_bg_iters = int(stage0_bg_iters)
    args.stage1_env_iters = int(stage1_env_iters)
    args.env_stage1_ignore_unfixed = _as_bool(_get(cfg, "schedule.env_stage1_ignore_unfixed", None), False)
    args.eval_first = eval_first

    # Unfixed background
    args.unfixed_gaussians = bool(unfixed_gaussians)
    args.unfixed_num_points = int(unfixed_num_points)
    args.unfixed_seed = int(unfixed_seed)
    args.unfixed_exclude_object_aabb_margin_ratio = float(unfixed_exclude_ratio)
    args.unfixed_disable_densification = bool(unfixed_disable_densification)
    args.unfixed_disable_opacity_reset = bool(unfixed_disable_opacity_reset)
    args.lambda_unfixed_obj_overlap = float(lambda_unfixed_obj_overlap)

    # Probe
    args.light_model = "probe" if light_object_model == "probe" else "envmap"
    args.probe_backend = probe_backend
    args.probe_dir_encoding = probe_dir_encoding
    args.probe_sh_degree = probe_sh_degree
    args.probe_fourier_n_frequencies = probe_fourier_n_freq
    args.probe_n_levels = probe_n_levels
    args.probe_n_features_per_level = probe_n_feat
    args.probe_log2_hashmap_size = probe_log2_hash
    args.probe_base_resolution = probe_base_res
    args.probe_per_level_scale = probe_per_level_scale
    args.probe_hidden_dim = probe_hidden_dim
    args.probe_n_hidden_layers = probe_n_hidden
    args.probe_output_activation = probe_out_act
    args.probe_output_softplus_beta = probe_out_beta
    args.probe_lr = probe_lr
    args.probe_weight_decay = probe_weight_decay
    args.probe_update_after = probe_update_after
    args.probe_update_interval = probe_update_interval
    args.freeze_probe = freeze_probe
    args.probe_pth = probe_pth
    args.probe_aabb_margin_ratio = probe_aabb_margin_ratio
    args.probe_aabb_min = probe_aabb_min
    args.probe_aabb_max = probe_aabb_max

    # Debug flags
    args.quiet = quiet
    args.debug_nonfinite_dump = debug_nonfinite_dump
    args.debug_nonfinite_dump_full = debug_nonfinite_dump_full
    args.debug_nonfinite_raise = debug_nonfinite_raise
    args.dump_env_map_on_eval = dump_env_map_on_eval
    args.tb_log_env_map = tb_log_env_map

    # Compatibility placeholders (not used by v2)
    args.ip = _as_str(_get(cfg, "runtime.gui_ip", None), "127.0.0.1")
    args.port = _as_int(_get(cfg, "runtime.gui_port", None), 6009)
    args.ground_plane_json = _get(cfg, "background.ground_plane.json", None)
    args.ground_texture = _get(cfg, "background.ground_plane.texture", None)

    return args


def _validate_args(args: Namespace) -> None:
    if not getattr(args, "model_path", None):
        raise ValueError("args.model_path is required")
    if not getattr(args, "source_path", None):
        raise ValueError("args.source_path is required")
    if not getattr(args, "gt_ply", None):
        raise ValueError("args.gt_ply is required")

    # Scene type detection requires either:
    # - COLMAP: <source_path>/sparse/
    # - Blender/NeRF synthetic: <source_path>/transforms_train.json
    src = str(args.source_path)
    has_colmap = os.path.exists(os.path.join(src, "sparse"))
    has_blender = os.path.exists(os.path.join(src, "transforms_train.json"))
    if not (has_colmap or has_blender):
        try:
            entries = sorted(os.listdir(src)) if os.path.isdir(src) else []
        except Exception:
            entries = []
        raise ValueError(
            "Could not recognize scene type from paths.source. "
            "Expected either '<source>/sparse/' (COLMAP) or '<source>/transforms_train.json' (Blender). "
            f"Got source_path={src!r}. "
            f"Top-level entries={entries[:30]!r}"
        )

    if getattr(args, "light_model", "envmap") == "probe" and getattr(args, "object_render_mode", "pbr") != "pbr":
        raise ValueError("--light_model=probe requires --object_render_mode=pbr")


def _write_resolved_config(model_path: str, cfg: dict[str, Any]) -> None:
    os.makedirs(model_path, exist_ok=True)
    out_path = os.path.join(model_path, "config_resolved.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    print(f"[V2] Wrote resolved config: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Static PBR training (v2 JSON config)")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    ap.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config keys with dotted paths: key=value (value parsed as JSON if possible).",
    )
    ap.add_argument("--print_config", action="store_true", help="Print resolved config and exit.")
    ap.add_argument("--dry_run", action="store_true", help="Validate + write resolved config then exit (no training).")

    cli = ap.parse_args()

    cfg = _load_json(cli.config)
    for ov in cli.override:
        if "=" not in ov:
            raise ValueError(f"Invalid --override (expected key=value): {ov}")
        k, raw_v = ov.split("=", 1)
        _set_by_dotted_path(cfg, k.strip(), _parse_override_value(raw_v.strip()))

    if cli.print_config:
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    args = _build_args_from_config(cfg)
    _validate_args(args)

    _write_resolved_config(args.model_path, cfg)
    if cli.dry_run:
        print("[V2] dry_run: exiting before training.")
        return 0

    # Match train_pbr.py main behavior for save/test schedules.
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    if args.test_interval and args.test_interval > 0:
        args.test_iterations = list(range(args.test_interval, args.iterations + 1, args.test_interval))
        if args.iterations not in args.test_iterations:
            args.test_iterations.append(args.iterations)
        args.test_iterations = sorted(set(args.test_iterations))

    # Build param groups using the existing engine classes.
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    opt = op.extract(args)
    opt.lambda_rgb = args.lambda_rgb
    opt.env_light_lr = args.env_light_lr
    opt.lambda_pbr = args.lambda_pbr
    opt.lambda_pbr_reg = args.lambda_pbr_reg
    opt.lambda_env_tv = args.lambda_env_tv
    opt.albedo_lr = args.albedo_lr
    opt.roughness_lr = args.roughness_lr
    opt.metallic_lr = args.metallic_lr

    training_pbr_static(lp.extract(args), opt, pp.extract(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
