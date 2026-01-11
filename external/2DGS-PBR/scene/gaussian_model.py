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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # PBR activation functions
        self.albedo_activation = torch.sigmoid
        self.roughness_activation = torch.sigmoid
        self.metallic_activation = torch.sigmoid


    def __init__(self, sh_degree: int, use_pbr: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.use_pbr = use_pbr  # PBR mode flag

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # PBR material parameters
        self._albedo = torch.empty(0)      # [N, 3] base color
        self._roughness = torch.empty(0)   # [N, 1] roughness 0-1
        self._metallic = torch.empty(0)    # [N, 1] metallic 0-1

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    # PBR property getters
    @property
    def get_albedo(self):
        """Get albedo (base color) constrained to [0, 1]"""
        return self.albedo_activation(self._albedo)

    @property
    def get_roughness(self):
        """Get roughness constrained to [0.1, 0.999] to avoid too smooth surfaces"""
        return torch.clamp(self.roughness_activation(self._roughness), min=0.1, max=0.999)

    @property
    def get_metallic(self):
        """Get metallic constrained to [0, 1]"""
        return self.metallic_activation(self._metallic)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert batch of rotation matrices to quaternions (w, x, y, z).
        R: [N, 3, 3]
        """
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        q = torch.zeros((R.shape[0], 4), device=R.device)

        # Case 1: tr > 0
        mask1 = tr > 0
        S1 = torch.sqrt(tr[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * S1
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S1
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S1
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S1

        # Case 2: tr <= 0
        mask_not1 = ~mask1
        # Find max diagonal element
        d = torch.diagonal(R[mask_not1], dim1=-2, dim2=-1)
        max_diag_idx = torch.argmax(d, dim=1)
        
        # Subcase 0: R[0,0] is max
        m0 = (max_diag_idx == 0)
        mask2 = torch.zeros_like(mask1)
        mask2[mask_not1] = m0
        
        if mask2.any():
            S2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
            q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S2
            q[mask2, 1] = 0.25 * S2
            q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S2
            q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S2

        # Subcase 1: R[1,1] is max
        m1 = (max_diag_idx == 1)
        mask3 = torch.zeros_like(mask1)
        mask3[mask_not1] = m1
        
        if mask3.any():
            S3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
            q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S3
            q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S3
            q[mask3, 2] = 0.25 * S3
            q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S3

        # Subcase 2: R[2,2] is max
        m2 = (max_diag_idx == 2)
        mask4 = torch.zeros_like(mask1)
        mask4[mask_not1] = m2
        
        if mask4.any():
            S4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
            q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S4
            q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S4
            q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S4
            q[mask4, 3] = 0.25 * S4

        return q

    def create_from_dense_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Initialize Gaussians from a dense point cloud with normals.
        - XYZ: set to points
        - Rotation: aligned with normals
        - Scale: estimated from K-NN density
        """
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        normals = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        print(f"Initializing from dense PCD with {points.shape[0]} points")

        # 1. Colors to SH
        fused_color = RGB2SH(colors)
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # 2. Scales from KNN (density)
        # Use mean distance to 3 nearest neighbors as base scale
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        # scales = [N, 2] for 2DGS. We use sqrt(dist) for both axes.
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)

        # 3. Rotations from Normals
        # 2DGS surfels are disks in XY plane (local normal = +Z)
        # We need rotation R s.t. R * [0,0,1] = target_normal
        
        # Normalize normals
        normals = torch.nn.functional.normalize(normals, dim=1)
        
        # Construct coordinate frame [x_axis, y_axis, z_axis=normal]
        z_axis = normals
        
        # Create arbitrary vector for cross product (avoid collinearity)
        # If normal is close to [1,0,0], use [0,1,0], else [1,0,0]
        ref_vec = torch.zeros_like(normals)
        ref_vec[:, 0] = 1.0
        mask = torch.abs(normals[:, 0]) > 0.9
        ref_vec[mask, 0] = 0.0
        ref_vec[mask, 1] = 1.0
        
        x_axis = torch.cross(ref_vec, z_axis, dim=1)
        x_axis = torch.nn.functional.normalize(x_axis, dim=1)
        
        y_axis = torch.cross(z_axis, x_axis, dim=1)
        y_axis = torch.nn.functional.normalize(y_axis, dim=1)
        
        # R = [x, y, z] columns
        R = torch.stack((x_axis, y_axis, z_axis), dim=2) # [N, 3, 3]
        
        # Convert to quaternion (w, x, y, z)
        rots = self._rotation_matrix_to_quaternion(R)

        # 4. Opacities
        # Initialize as semi-opaque
        opacities = self.inverse_opacity_activation(0.5 * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda"))

        # Set parameters
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Initialize PBR parameters
        if self.use_pbr:
            self._albedo = nn.Parameter(torch.logit(colors.clamp(0.01, 0.99)).requires_grad_(True))
            self._roughness = nn.Parameter(torch.full((points.shape[0], 1), 0.5, device="cuda").requires_grad_(True))
            self._metallic = nn.Parameter(torch.full((points.shape[0], 1), -2.0, device="cuda").requires_grad_(True))

    def training_setup_fixed_geometry(self, training_args):
        """
        Setup optimizer for Fixed Geometry training.
        - Unlocked: Scaling, Opacity, SH (Color), PBR (Albedo, Roughness, Metallic)
        - Locked: XYZ, Rotation
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Note: xyz and rotation are NOT in this list
        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        ]

        if self.use_pbr:
            albedo_lr = getattr(training_args, 'albedo_lr', 0.001)
            roughness_lr = getattr(training_args, 'roughness_lr', 0.0002)
            metallic_lr = getattr(training_args, 'metallic_lr', 0.0002)

            l.extend([
                {'params': [self._albedo], 'lr': albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': metallic_lr, "name": "metallic"},
            ])
            print(f"Fixed Geometry (Scale Optimized) - PBR Optimizer: albedo_lr={albedo_lr}, roughness_lr={roughness_lr}, metallic_lr={metallic_lr}")

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # Dummy scheduler for XYZ since we don't optimize it, but existing code might call it
        self.xyz_scheduler_args = lambda x: 0.0

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        num_pts = fused_point_cloud.shape[0]

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Initialize PBR parameters
        if self.use_pbr:
            # Initialize albedo from point cloud colors (inverse sigmoid)
            colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            self._albedo = nn.Parameter(
                torch.logit(colors.clamp(0.01, 0.99)).requires_grad_(True)
            )
            # Initialize roughness to 0.6 (moderately rough), inverse sigmoid
            # sigmoid(0.4) ≈ 0.6
            self._roughness = nn.Parameter(
                torch.full((num_pts, 1), 0.4, device="cuda").requires_grad_(True)
            )
            # Initialize metallic to low value (most objects are non-metallic)
            # sigmoid(-2) ≈ 0.12
            self._metallic = nn.Parameter(
                torch.full((num_pts, 1), -2.0, device="cuda").requires_grad_(True)
            )
            print(f"PBR mode enabled: albedo {self._albedo.shape}, roughness {self._roughness.shape}, metallic {self._metallic.shape}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # Add PBR parameters to optimizer
        if self.use_pbr:
            # Get learning rates from training_args or use defaults
            albedo_lr = getattr(training_args, 'albedo_lr', 0.001)
            roughness_lr = getattr(training_args, 'roughness_lr', 0.0002)
            metallic_lr = getattr(training_args, 'metallic_lr', 0.0002)

            l.extend([
                {'params': [self._albedo], 'lr': albedo_lr, "name": "albedo"},
                {'params': [self._roughness], 'lr': roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': metallic_lr, "name": "metallic"},
            ])
            print(f"PBR optimizer: albedo_lr={albedo_lr}, roughness_lr={roughness_lr}, metallic_lr={metallic_lr}")

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        # PBR attributes
        if self.use_pbr and self._albedo.shape[0] > 0:
            for i in range(3):  # albedo RGB
                l.append('albedo_{}'.format(i))
            l.append('roughness')
            l.append('metallic')

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # Build attributes list
        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        # Add PBR attributes
        if self.use_pbr and self._albedo.shape[0] > 0:
            albedo = self._albedo.detach().cpu().numpy()
            roughness = self._roughness.detach().cpu().numpy()
            metallic = self._metallic.detach().cpu().numpy()
            attributes.extend([albedo, roughness, metallic])

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # Load PBR parameters if available and use_pbr is enabled
        if self.use_pbr:
            # Check if PBR attributes exist in the ply file
            property_names = [p.name for p in plydata.elements[0].properties]
            has_pbr = 'albedo_0' in property_names

            if has_pbr:
                albedo = np.stack((
                    np.asarray(plydata.elements[0]["albedo_0"]),
                    np.asarray(plydata.elements[0]["albedo_1"]),
                    np.asarray(plydata.elements[0]["albedo_2"])
                ), axis=1)
                roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
                metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

                self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
                self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
                self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
                print(f"Loaded PBR parameters: albedo {self._albedo.shape}, roughness {self._roughness.shape}, metallic {self._metallic.shape}")
            else:
                # Initialize PBR parameters with defaults if not in ply file
                num_pts = xyz.shape[0]
                self._albedo = nn.Parameter(torch.zeros((num_pts, 3), device="cuda").requires_grad_(True))
                self._roughness = nn.Parameter(torch.full((num_pts, 1), 0.4, device="cuda").requires_grad_(True))
                self._metallic = nn.Parameter(torch.full((num_pts, 1), -2.0, device="cuda").requires_grad_(True))
                print(f"PBR attributes not found in ply, initialized with defaults")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Prune PBR parameters
        if self.use_pbr and "albedo" in optimizable_tensors:
            self._albedo = optimizable_tensors["albedo"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                               new_albedo=None, new_roughness=None, new_metallic=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # Add PBR parameters to densification
        if self.use_pbr and new_albedo is not None:
            d["albedo"] = new_albedo
            d["roughness"] = new_roughness
            d["metallic"] = new_metallic

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update PBR parameters
        if self.use_pbr and "albedo" in optimizable_tensors:
            self._albedo = optimizable_tensors["albedo"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # PBR parameters
        new_albedo = None
        new_roughness = None
        new_metallic = None
        if self.use_pbr and self._albedo.shape[0] > 0:
            new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
            new_metallic = self._metallic[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_albedo, new_roughness, new_metallic)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # PBR parameters
        new_albedo = None
        new_roughness = None
        new_metallic = None
        if self.use_pbr and self._albedo.shape[0] > 0:
            new_albedo = self._albedo[selected_pts_mask]
            new_roughness = self._roughness[selected_pts_mask]
            new_metallic = self._metallic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_albedo, new_roughness, new_metallic)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1