"""
Mesh cleaning utilities for OpenMaterial evaluation

Adapted from: https://github.com/xxlong0/SparseNeuS/blob/main/evaluation/clean_mesh.py
"""

import numpy as np
import cv2 as cv
import os
from glob import glob
import trimesh
from pathlib import Path
import json
import math


def gen_w2c(pose):
    """Generate world-to-camera transformation"""
    pose[:3, :1] = -pose[:3, :1]
    pose[:3, 1:2] = -pose[:3, 1:2]  # Flip the x+ and y+ to align coordinate system

    R = pose[:3, :3].transpose()
    T = -R @ pose[:3, 3:]
    return R, T


def gen_camera_intrinsic(width, height, fov_x, fov_y):
    """Generate camera intrinsic matrix"""
    fx = width / 2.0 / math.tan(fov_x / 180 * math.pi / 2.0)
    fy = height / 2.0 / math.tan(fov_y / 180 * math.pi / 2.0)
    return fx, fy


def clean_points_by_mask(points, transforms_path, mask_dir, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    """
    Clean points by checking visibility in training masks

    Args:
        points: Nx3 array of 3D points
        transforms_path: Path to transforms_train.json
        mask_dir: Directory containing mask images
        imgs_idx: List of image indices to use (None for all)
        minimal_vis: Minimum number of views a point must be visible in
        mask_dilated_size: Size of dilation kernel for masks

    Returns:
        Boolean mask of valid points
    """
    with open(transforms_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fov_x = 37.8492
    fov_y = 28.8415
    width, height = 1600, 1200

    # Transform to Colmap format
    fx, fy = gen_camera_intrinsic(width, height, fov_x, fov_y)

    # Use float64 to avoid loss of precision
    intrinsic = np.diag([fx, fy, 1.0, 1.0]).astype(np.float64)
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0

    flip_mat = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0])

    mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
    n_images = len(mask_files)
    inside_mask = np.zeros(len(points))

    if imgs_idx is None:
        imgs_idx = list(range(n_images))

    for i, frame in enumerate(data['frames']):
        if i >= len(mask_files):
            break

        cam_pose_ = np.matmul(frame['transform_matrix'], flip_mat)
        cam_pose = np.array(cam_pose_)
        R, T = gen_w2c(cam_pose)
        w2c = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
        world_mat = intrinsic @ w2c

        P = world_mat @ scale_mat
        P = P[:3, :4]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_files[i])
        kernel_size = mask_dilated_size
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        in_mask = (pts_image[:, 0] >= 0) * (pts_image[:, 0] <= 1600) * (pts_image[:, 1] >= 0) * (
                pts_image[:, 1] <= 1200) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]
        curr_mask = curr_mask.astype(np.float32) * in_mask

        inside_mask += curr_mask

        if i >= len(imgs_idx):
            break

    return inside_mask > minimal_vis


def clean_mesh_by_mask(mesh_file, output_file, transforms_path, mask_dir, cut_y=-1.0, minimal_vis=2, mask_dilated_size=11):
    """
    Clean mesh by removing faces not visible in training masks

    Args:
        mesh_file: Input mesh file path
        output_file: Output cleaned mesh file path
        transforms_path: Path to transforms_train.json
        mask_dir: Directory containing mask images
        cut_y: Y coordinate threshold for cutting bottom (from GT mesh bbox)
        minimal_vis: Minimum visibility count
        mask_dilated_size: Mask dilation kernel size
    """
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]

    # Clean by mask visibility
    mask = clean_points_by_mask(old_vertices, transforms_path, mask_dir, None, minimal_vis, mask_dilated_size)

    # Cut by Y coordinate (remove bottom)
    y_mask = old_vertices[:, 1] >= cut_y
    mask = mask & y_mask

    # Build vertex index mapping
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    # Filter faces
    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    # Create and save new mesh
    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    new_mesh.export(output_file)

    return output_file


def clean_points_by_visualhull(points, transforms_path, mask_dir, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    """
    Clean points by visual hull (remove points outside all masks)

    Args:
        points: Nx3 array of 3D points
        transforms_path: Path to transforms_train.json
        mask_dir: Directory containing mask images
        imgs_idx: List of image indices to use (None for all)
        minimal_vis: Minimum number of views a point must be outside
        mask_dilated_size: Size of dilation kernel for masks (default: 101 in original)

    Returns:
        Tuple of (boolean mask of valid points, scale_mat)
    """
    with open(transforms_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fov_x = 37.8492
    fov_y = 28.8415
    width, height = 1600, 1200

    # Transform to Colmap format
    fx, fy = gen_camera_intrinsic(width, height, fov_x, fov_y)

    # Use float64 to avoid loss of precision
    intrinsic = np.diag([fx, fy, 1.0, 1.0]).astype(np.float64)
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0

    flip_mat = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0])

    mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
    n_images = len(mask_files)
    outside_mask = np.zeros(len(points))

    if imgs_idx is None:
        imgs_idx = list(range(n_images))

    for i in imgs_idx:
        if i >= len(data['frames']) or i >= len(mask_files):
            break

        cam_pose_ = np.matmul(data['frames'][i]['transform_matrix'], flip_mat)
        cam_pose = np.array(cam_pose_)
        R, T = gen_w2c(cam_pose)
        w2c = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
        world_mat = intrinsic @ w2c

        P = world_mat @ scale_mat
        P = P[:3, :4]

        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_files[i])
        kernel_size = mask_dilated_size
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] < 128)  # OUTSIDE the mask

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        border = 50
        in_mask = (pts_image[:, 0] >= (0 + border)) * (pts_image[:, 0] <= (1600 - border)) * (
                pts_image[:, 1] >= (0 + border)) * (pts_image[:, 1] <= (1200 - border)) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        curr_mask = curr_mask.astype(np.float32) * in_mask
        outside_mask += curr_mask

    return outside_mask < 5, scale_mat


def clean_mesh_by_visualhull(mesh_file, output_file, transforms_path, mask_dir, minimal_vis=2, mask_dilated_size=31):
    """
    Clean mesh by visual hull (second stage cleaning with larger dilation)

    Args:
        mesh_file: Input mesh file path
        output_file: Output cleaned mesh file path
        transforms_path: Path to transforms_train.json
        mask_dir: Directory containing mask images
        minimal_vis: Minimum visibility count
        mask_dilated_size: Mask dilation kernel size (default: 31 = 11+20)
    """
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]

    mask, scale_mat = clean_points_by_visualhull(old_vertices, transforms_path, mask_dir, None, minimal_vis, mask_dilated_size)

    # Build vertex index mapping
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    # Filter faces
    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    # Create new mesh
    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    new_mesh.vertices *= scale_mat[0, 0]
    new_mesh.vertices += scale_mat[:3, 3]

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    new_mesh.export(output_file)

    return output_file
