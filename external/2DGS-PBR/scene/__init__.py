#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path  # Store for pseudo-GT loading
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.gaussians._xyz.shape[0] == 0:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def load_pseudo_gt_cache(self, depth_subdir: str = 'depth', normal_subdir: str = 'normal'):
        """
        Preload all pseudo-GT depth and normal maps into Camera objects.

        This eliminates per-iteration disk I/O during training.
        Data is stored on CPU and moved to CUDA when needed.

        Args:
            depth_subdir: Subdirectory name for depth maps (default: 'depth')
            normal_subdir: Subdirectory name for normal maps (default: 'normal')
        """
        depth_root = os.path.join(self.source_path, depth_subdir)
        normal_root = os.path.join(self.source_path, normal_subdir)

        depth_exists = os.path.exists(depth_root)
        normal_exists = os.path.exists(normal_root)

        if not depth_exists and not normal_exists:
            print(f"[Warning] Neither depth ({depth_root}) nor normal ({normal_root}) directories found.")
            return

        print(f"Preloading pseudo-GT data...")
        if depth_exists:
            print(f"  Depth path: {depth_root}")
        if normal_exists:
            print(f"  Normal path: {normal_root}")

        # Load for all resolution scales
        for scale in self.train_cameras.keys():
            cameras = self.train_cameras[scale]
            for cam in tqdm(cameras, desc=f"Loading pseudo-GT (scale={scale})"):
                self._load_single_pseudo_gt(cam, depth_root, normal_root, depth_exists, normal_exists)

            # Also load for test cameras if needed
            test_cameras = self.test_cameras.get(scale, [])
            if test_cameras:
                for cam in tqdm(test_cameras, desc=f"Loading test pseudo-GT (scale={scale})"):
                    self._load_single_pseudo_gt(cam, depth_root, normal_root, depth_exists, normal_exists)

        # Count loaded
        n_depth = sum(1 for cam in self.train_cameras[1.0] if cam.pseudo_gt_depth is not None)
        n_normal = sum(1 for cam in self.train_cameras[1.0] if cam.pseudo_gt_normal is not None)
        print(f"Pseudo-GT loaded: {n_depth} depth maps, {n_normal} normal maps")

    def _load_single_pseudo_gt(self, cam, depth_root, normal_root, depth_exists, normal_exists):
        """Load pseudo-GT for a single camera and store in Camera object."""
        filename = cam.image_name

        # Load Depth
        if depth_exists:
            d_path = os.path.join(depth_root, f"{filename}.png")
            if os.path.exists(d_path):
                depth_img = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    # Convert to float tensor (keep on CPU)
                    depth_tensor = torch.from_numpy(depth_img.astype(np.float32))

                    # Resize if needed
                    if depth_tensor.shape[:2] != (cam.image_height, cam.image_width):
                        depth_tensor = TF.resize(
                            depth_tensor.unsqueeze(0),
                            (cam.image_height, cam.image_width)
                        ).squeeze(0)

                    cam.pseudo_gt_depth = depth_tensor.unsqueeze(0)  # [1, H, W]

        # Load Normal
        if normal_exists:
            n_path = os.path.join(normal_root, f"{filename}.png")
            if os.path.exists(n_path):
                normal_img = cv2.imread(n_path)
                if normal_img is not None:
                    # BGR -> RGB
                    normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
                    # Convert to float tensor [0, 1] -> [-1, 1]
                    normal_tensor = torch.from_numpy(normal_img.astype(np.float32) / 255.0)
                    normal_tensor = normal_tensor * 2.0 - 1.0
                    # [H, W, 3] -> [3, H, W]
                    normal_tensor = normal_tensor.permute(2, 0, 1)

                    # Resize if needed
                    if normal_tensor.shape[1:] != (cam.image_height, cam.image_width):
                        normal_tensor = TF.resize(
                            normal_tensor,
                            (cam.image_height, cam.image_width)
                        )

                    cam.pseudo_gt_normal = normal_tensor  # [3, H, W]