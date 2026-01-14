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
import matplotlib.pyplot as plt
import torch.nn.functional as F

def _expand_mask_to_channels(mask: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """
    Expand a single-channel mask to match the channel dimension of an image tensor.

    Supported:
    - img: [C, H, W], mask: [1, H, W] -> [C, H, W]
    - img: [B, C, H, W], mask: [B, 1, H, W] -> [B, C, H, W]
    - img: [B, C, H, W], mask: [1, H, W] -> [1, C, H, W] (broadcastable to B)
    """
    if mask is None:
        return None

    if img.dim() == 3 and mask.dim() == 3:
        if mask.shape[0] == 1 and img.shape[0] != 1:
            return mask.expand(img.shape[0], -1, -1)
        return mask

    if img.dim() == 4:
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.dim() == 4 and mask.shape[1] == 1 and img.shape[1] != 1:
            return mask.expand(-1, img.shape[1], -1, -1)
        return mask

    return mask

def mse(img1, img2, mask=None):
    diff_sq = ((img1 - img2)) ** 2
    if mask is not None:
        mask = _expand_mask_to_channels(mask, diff_sq)
        return (diff_sq * mask).sum() / (mask.sum() + 1e-8)
    return diff_sq.reshape(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    diff_sq = ((img1 - img2)) ** 2
    if mask is not None:
        mask = _expand_mask_to_channels(mask, diff_sq)
        mse_val = (diff_sq * mask).sum() / (mask.sum() + 1e-8)
    else:
        mse_val = diff_sq.reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_val + 1e-8))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image
