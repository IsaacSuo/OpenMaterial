"""
Evaluation utilities for OpenMaterial benchmark
"""

from .clean_mesh import (
    clean_mesh_by_mask,
    clean_points_by_mask,
    clean_mesh_by_visualhull,
    clean_points_by_visualhull
)

__all__ = [
    'clean_mesh_by_mask',
    'clean_points_by_mask',
    'clean_mesh_by_visualhull',
    'clean_points_by_visualhull'
]
