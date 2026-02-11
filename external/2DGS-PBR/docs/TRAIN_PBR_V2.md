# Train PBR v2 (JSON Config + CLI Overrides)

`train_pbr_v2.py` is a config-driven entrypoint for the static-geometry PBR pipeline.

It keeps the training “engine” in `train_pbr.py:training_pbr_static`, but replaces the long CLI with:

- `--config <json>`
- `--override key=value` (repeatable; value parsed as JSON if possible)

## Quickstart

```bash
python train_pbr_v2.py --config configs/pbr_env_probe_unfixed_complex.json
```

Print the resolved config and exit:

```bash
python train_pbr_v2.py --config configs/pbr_env_probe_unfixed_complex.json --print_config
```

Validate + write `config_resolved.json` then exit (no training):

```bash
python train_pbr_v2.py --config configs/pbr_env_probe_unfixed_complex.json --dry_run
```

## Common Overrides

All overrides are dotted paths into the JSON config.

Examples:

```bash
# Change total iterations
--override optim.iters=60000

# Make pruning less aggressive (keep more background gaussians)
--override optim.opacity_cull=0.01

# Change evaluation frequency
--override schedule.eval.every=1000

# Tune probe learning rate (object lighting)
--override lighting.object.probe.lr=0.005

# Increase unfixed overlap penalty (discourage background gaussians covering object)
--override background.unfixed.lambda_obj_overlap=0.05

# Freeze env map (background stays fixed; object uses probe)
--override lighting.sky.envmap.freeze=true
```

## Mental Model (pbr + env + probe, Scheme A)

For the “A scheme” setup:

- Background uses `env_map` (EnvironmentLight) + optional `unfixed_gaussians` for finite-depth background geometry.
- Object shading uses PBR G-buffer + **probe** (position+direction -> HDR radiance).

Config keys:

- `object.mode="pbr"`
- `lighting.sky.model="envmap"`
- `lighting.object.model="probe"`
- `background.model="unfixed"` (optional but recommended for complex backgrounds)

## Outputs

`train_pbr_v2.py` writes the resolved config to:

- `<output>/config_resolved.json`

The training engine writes the usual artifacts under `<output>/` (point cloud, env_light ckpts, etc.).
