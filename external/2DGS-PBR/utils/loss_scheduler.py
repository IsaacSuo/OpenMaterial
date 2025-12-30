"""
Loss Scheduler for 2DGS-PBR Training

Manages the staged activation of losses during training with configurable
iteration boundaries and optional warm-up ramps.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrainingStages:
    """Configurable iteration boundaries for training stages."""
    # Stage 1: Geometry foundation (recon only)
    stage1_end: int = 3000

    # Stage 2: Geometric refinement (+ dist, mono supervision)
    stage2_end: int = 7000

    # Stage 3: PBR bootstrap (+ normal, pbr, env_tv)
    stage3_end: int = 15000

    # Stage 4: Full PBR (+ pbr_reg, fade mono)
    # Continues until end of training

    # Warm-up iterations for smooth loss activation
    warmup_iters: int = 1000

    # Whether to fade out mono supervision in stage 4
    fade_mono: bool = True
    fade_mono_end: int = 20000  # Fully faded by this iteration


@dataclass
class LossWeights:
    """Base loss weights (before scheduling)."""
    # Reconstruction
    lambda_dssim: float = 0.2

    # Geometry
    lambda_dist: float = 0.0  # From opt
    lambda_normal: float = 0.05  # From opt

    # Mono supervision
    lambda_mono_depth: float = 0.1
    lambda_mono_normal: float = 0.05

    # PBR
    lambda_pbr: float = 0.1
    lambda_pbr_reg: float = 0.01
    lambda_env_tv: float = 0.001


class LossScheduler:
    """
    Manages loss activation and weight scheduling throughout training.

    Training Stages:
        Stage 1 (0 - stage1_end): Geometry Foundation
            - L_recon only

        Stage 2 (stage1_end - stage2_end): Geometric Refinement
            - + L_dist (with warmup)
            - + L_mono_depth (if enabled, with warmup)
            - + L_mono_normal (if enabled, with warmup)

        Stage 3 (stage2_end - stage3_end): PBR Bootstrap
            - + L_normal (with warmup)
            - + L_pbr (with warmup)
            - + L_env_tv (with warmup)

        Stage 4 (stage3_end - end): Full PBR
            - + L_pbr_reg (with warmup)
            - Optionally fade out L_mono
    """

    def __init__(
        self,
        stages: Optional[TrainingStages] = None,
        base_weights: Optional[LossWeights] = None,
        use_mono_supervision: bool = False
    ):
        self.stages = stages or TrainingStages()
        self.base = base_weights or LossWeights()
        self.use_mono = use_mono_supervision

    def get_stage(self, iteration: int) -> int:
        """Get current training stage (1-4)."""
        if iteration < self.stages.stage1_end:
            return 1
        elif iteration < self.stages.stage2_end:
            return 2
        elif iteration < self.stages.stage3_end:
            return 3
        else:
            return 4

    def _warmup_factor(self, iteration: int, start_iter: int) -> float:
        """
        Calculate warmup factor for smooth loss activation.
        Returns 0.0 before start_iter, ramps to 1.0 over warmup_iters.
        """
        if iteration < start_iter:
            return 0.0

        elapsed = iteration - start_iter
        if elapsed >= self.stages.warmup_iters:
            return 1.0

        # Smooth cosine warmup
        import math
        return 0.5 * (1 - math.cos(math.pi * elapsed / self.stages.warmup_iters))

    def _fade_factor(self, iteration: int, start_iter: int, end_iter: int) -> float:
        """
        Calculate fade-out factor.
        Returns 1.0 before start_iter, fades to 0.0 by end_iter.
        """
        if iteration < start_iter:
            return 1.0
        if iteration >= end_iter:
            return 0.0

        import math
        progress = (iteration - start_iter) / (end_iter - start_iter)
        return 0.5 * (1 + math.cos(math.pi * progress))

    def get_weights(self, iteration: int) -> Dict[str, float]:
        """
        Get all loss weights for current iteration.

        Returns:
            Dict with keys: 'dist', 'normal', 'mono_depth', 'mono_normal',
                           'pbr', 'pbr_reg', 'env_tv'
        """
        weights = {}

        # Stage 2+: dist loss
        if iteration >= self.stages.stage1_end:
            warmup = self._warmup_factor(iteration, self.stages.stage1_end)
            weights['dist'] = self.base.lambda_dist * warmup
        else:
            weights['dist'] = 0.0

        # Stage 2+: mono supervision (if enabled)
        if self.use_mono and iteration >= self.stages.stage1_end:
            warmup = self._warmup_factor(iteration, self.stages.stage1_end)

            # Apply fade in stage 4 if configured
            fade = 1.0
            if self.stages.fade_mono and iteration >= self.stages.stage3_end:
                fade = self._fade_factor(
                    iteration,
                    self.stages.stage3_end,
                    self.stages.fade_mono_end
                )

            weights['mono_depth'] = self.base.lambda_mono_depth * warmup * fade
            weights['mono_normal'] = self.base.lambda_mono_normal * warmup * fade
        else:
            weights['mono_depth'] = 0.0
            weights['mono_normal'] = 0.0

        # Stage 3+: normal consistency
        if iteration >= self.stages.stage2_end:
            warmup = self._warmup_factor(iteration, self.stages.stage2_end)
            weights['normal'] = self.base.lambda_normal * warmup
        else:
            weights['normal'] = 0.0

        # Stage 3+: PBR shading + env TV
        if iteration >= self.stages.stage2_end:
            warmup = self._warmup_factor(iteration, self.stages.stage2_end)
            weights['pbr'] = self.base.lambda_pbr * warmup
            weights['env_tv'] = self.base.lambda_env_tv * warmup
        else:
            weights['pbr'] = 0.0
            weights['env_tv'] = 0.0

        # Stage 4: PBR regularization
        if iteration >= self.stages.stage3_end:
            warmup = self._warmup_factor(iteration, self.stages.stage3_end)
            weights['pbr_reg'] = self.base.lambda_pbr_reg * warmup
        else:
            weights['pbr_reg'] = 0.0

        return weights

    def is_pbr_active(self, iteration: int) -> bool:
        """Check if PBR shading should be computed."""
        return iteration >= self.stages.stage2_end

    def is_pbr_reg_active(self, iteration: int) -> bool:
        """Check if PBR regularization should be computed."""
        return iteration >= self.stages.stage3_end

    def should_optimize_env_light(self, iteration: int) -> bool:
        """Check if environment light should be optimized."""
        return iteration >= self.stages.stage2_end

    def get_stage_info(self, iteration: int) -> str:
        """Get human-readable stage description."""
        stage = self.get_stage(iteration)
        descriptions = {
            1: "Stage 1: Geometry Foundation (recon only)",
            2: "Stage 2: Geometric Refinement (+ dist, mono)",
            3: "Stage 3: PBR Bootstrap (+ normal, pbr, env_tv)",
            4: "Stage 4: Full PBR (+ pbr_reg)"
        }
        return descriptions.get(stage, "Unknown stage")

    def print_config(self):
        """Print training stage configuration."""
        print("\n" + "=" * 60)
        print("Loss Scheduler Configuration")
        print("=" * 60)
        print(f"Stage 1 (Geometry Foundation):    0 - {self.stages.stage1_end}")
        print(f"  Active: L_recon")
        print(f"Stage 2 (Geometric Refinement):   {self.stages.stage1_end} - {self.stages.stage2_end}")
        print(f"  Active: + L_dist" + (", L_mono_depth, L_mono_normal" if self.use_mono else ""))
        print(f"Stage 3 (PBR Bootstrap):          {self.stages.stage2_end} - {self.stages.stage3_end}")
        print(f"  Active: + L_normal, L_pbr, L_env_tv")
        print(f"Stage 4 (Full PBR):               {self.stages.stage3_end} - end")
        print(f"  Active: + L_pbr_reg" + (", fade L_mono" if self.stages.fade_mono and self.use_mono else ""))
        print(f"\nWarmup iterations: {self.stages.warmup_iters}")
        if self.use_mono and self.stages.fade_mono:
            print(f"Mono fade end: {self.stages.fade_mono_end}")
        print("=" * 60 + "\n")

    @classmethod
    def from_args(cls, opt) -> 'LossScheduler':
        """Create LossScheduler from parsed arguments."""
        stages = TrainingStages(
            stage1_end=getattr(opt, 'stage1_end', 3000),
            stage2_end=getattr(opt, 'stage2_end', 7000),
            stage3_end=getattr(opt, 'stage3_end', 15000),
            warmup_iters=getattr(opt, 'warmup_iters', 1000),
            fade_mono=getattr(opt, 'fade_mono', True),
            fade_mono_end=getattr(opt, 'fade_mono_end', 20000),
        )

        base_weights = LossWeights(
            lambda_dssim=getattr(opt, 'lambda_dssim', 0.2),
            lambda_dist=getattr(opt, 'lambda_dist', 0.0),
            lambda_normal=getattr(opt, 'lambda_normal', 0.05),
            lambda_mono_depth=getattr(opt, 'lambda_mono_depth', 0.1),
            lambda_mono_normal=getattr(opt, 'lambda_mono_normal', 0.05),
            lambda_pbr=getattr(opt, 'lambda_pbr', 0.1),
            lambda_pbr_reg=getattr(opt, 'lambda_pbr_reg', 0.01),
            lambda_env_tv=getattr(opt, 'lambda_env_tv', 0.001),
        )

        return cls(
            stages=stages,
            base_weights=base_weights,
            use_mono_supervision=getattr(opt, 'use_pseudo_gt', False)
        )
