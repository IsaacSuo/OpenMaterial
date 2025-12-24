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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ==================== PBR Loss Functions ====================

def pbr_reconstruction_loss(shaded_image, gt_image, lambda_l1=0.8, lambda_ssim=0.2):
    """
    PBR reconstruction loss combining L1 and SSIM.

    Args:
        shaded_image: [3, H, W] PBR-shaded output
        gt_image: [3, H, W] ground truth image
        lambda_l1: weight for L1 loss
        lambda_ssim: weight for SSIM loss

    Returns:
        Combined reconstruction loss
    """
    l1 = l1_loss(shaded_image, gt_image)
    ssim_val = ssim(shaded_image.unsqueeze(0), gt_image.unsqueeze(0))
    return lambda_l1 * l1 + lambda_ssim * (1.0 - ssim_val)


def material_smoothness_loss(material_map, alpha_map=None):
    """
    Encourage spatial smoothness of material properties.
    Uses total variation (TV) loss weighted by alpha (where visible).

    Args:
        material_map: [C, H, W] material property map
        alpha_map: [1, H, W] optional alpha/opacity map for weighting

    Returns:
        Smoothness loss value
    """
    # Total variation: sum of absolute differences
    tv_h = torch.abs(material_map[:, 1:, :] - material_map[:, :-1, :])
    tv_w = torch.abs(material_map[:, :, 1:] - material_map[:, :, :-1])

    if alpha_map is not None:
        # Weight by alpha (only regularize visible regions)
        alpha_h = (alpha_map[:, 1:, :] + alpha_map[:, :-1, :]) / 2
        alpha_w = (alpha_map[:, :, 1:] + alpha_map[:, :, :-1]) / 2
        tv_h = tv_h * alpha_h
        tv_w = tv_w * alpha_w

    return tv_h.mean() + tv_w.mean()


def metallic_prior_loss(metallic_map, alpha_map=None):
    """
    Encourage metallic to be binary (0 or 1).
    Most real materials are either dielectric (0) or metallic (1).

    Uses: min(metallic, 1 - metallic) to encourage values near 0 or 1.

    Args:
        metallic_map: [1, H, W] metallic map in [0, 1]
        alpha_map: [1, H, W] optional alpha for weighting

    Returns:
        Prior loss value
    """
    # Distance to nearest binary value
    binary_dist = torch.min(metallic_map, 1.0 - metallic_map)

    if alpha_map is not None:
        binary_dist = binary_dist * alpha_map

    return binary_dist.mean()


def roughness_prior_loss(roughness_map, alpha_map=None, target_roughness=0.5):
    """
    Soft regularization on roughness to prevent extreme values.
    Encourages roughness to stay in reasonable range.

    Args:
        roughness_map: [1, H, W] roughness map in [0.1, 0.999]
        alpha_map: [1, H, W] optional alpha for weighting
        target_roughness: target roughness value (default 0.5)

    Returns:
        Prior loss value
    """
    # L2 deviation from target
    deviation = (roughness_map - target_roughness) ** 2

    if alpha_map is not None:
        deviation = deviation * alpha_map

    return deviation.mean()


def albedo_chroma_loss(albedo_map, alpha_map=None):
    """
    Encourage albedo to have consistent chromaticity.
    This helps prevent color bleeding and unrealistic color variations.

    Args:
        albedo_map: [3, H, W] albedo map
        alpha_map: [1, H, W] optional alpha for weighting

    Returns:
        Chroma consistency loss
    """
    # Compute luminance
    luminance = 0.2126 * albedo_map[0:1] + 0.7152 * albedo_map[1:2] + 0.0722 * albedo_map[2:3]

    # Compute chromaticity (color / luminance)
    chroma = albedo_map / (luminance + 1e-6)

    # TV loss on chromaticity (encourage smooth chroma)
    tv_h = torch.abs(chroma[:, 1:, :] - chroma[:, :-1, :])
    tv_w = torch.abs(chroma[:, :, 1:] - chroma[:, :, :-1])

    if alpha_map is not None:
        alpha_h = (alpha_map[:, 1:, :] + alpha_map[:, :-1, :]) / 2
        alpha_w = (alpha_map[:, :, 1:] + alpha_map[:, :, :-1]) / 2
        tv_h = tv_h * alpha_h
        tv_w = tv_w * alpha_w

    return tv_h.mean() + tv_w.mean()


def compute_pbr_losses(
    gbuffer_albedo,
    gbuffer_roughness,
    gbuffer_metallic,
    alpha_map=None,
    lambda_albedo_smooth=0.01,
    lambda_roughness_smooth=0.01,
    lambda_metallic_smooth=0.01,
    lambda_metallic_prior=0.001,
    lambda_roughness_prior=0.001,
    lambda_albedo_chroma=0.001,
):
    """
    Compute all PBR material regularization losses.

    Args:
        gbuffer_albedo: [3, H, W] albedo map
        gbuffer_roughness: [1, H, W] roughness map
        gbuffer_metallic: [1, H, W] metallic map
        alpha_map: [1, H, W] optional alpha map
        lambda_*: loss weights

    Returns:
        dict with individual losses and total
    """
    losses = {}

    # Smoothness losses
    losses['albedo_smooth'] = lambda_albedo_smooth * material_smoothness_loss(
        gbuffer_albedo, alpha_map
    )
    losses['roughness_smooth'] = lambda_roughness_smooth * material_smoothness_loss(
        gbuffer_roughness, alpha_map
    )
    losses['metallic_smooth'] = lambda_metallic_smooth * material_smoothness_loss(
        gbuffer_metallic, alpha_map
    )

    # Prior losses
    losses['metallic_prior'] = lambda_metallic_prior * metallic_prior_loss(
        gbuffer_metallic, alpha_map
    )
    losses['roughness_prior'] = lambda_roughness_prior * roughness_prior_loss(
        gbuffer_roughness, alpha_map
    )

    # Albedo chroma consistency
    losses['albedo_chroma'] = lambda_albedo_chroma * albedo_chroma_loss(
        gbuffer_albedo, alpha_map
    )

    # Total
    losses['total_pbr_reg'] = sum(losses.values())

    return losses

