#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import os.path
from typing import Literal, Dict, Any, Tuple, List

import drjit as dr
import mitsuba as mi
import torch
import trimesh
import trimesh.creation
import trimesh.transformations
import trimesh.util
from rich.console import Console

from emitters.nerf_op import get_ray_bundle
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox, CropMode
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.colormaps import linear_to_srgb

CONSOLE = Console(width=120)


def generate_rays_spherical(width: int, height: int, radius: float = 1.0, inward=True, device='cpu',
                            camera_idx: int = 0):
    theta_range = torch.linspace(0, torch.pi, height, device=device)
    phi_range = torch.linspace(0, torch.pi * 2, width, device=device)
    phi, theta = torch.meshgrid(phi_range, theta_range, indexing='xy')
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    xyz = torch.stack([x, y, z], dim=-1)
    directions = -xyz if inward else xyz
    positions = xyz * radius
    return RayBundle(
        origins=positions,
        directions=directions,
        pixel_area=torch.ones_like(positions[..., :1]),
        camera_indices=torch.full_like(positions[..., :1], fill_value=camera_idx, dtype=torch.long),
    )


def update_bbox_near_far(ray_bundle: RayBundle, bound_r: float):
    if ray_bundle.nears is None:
        ray_bundle.nears = torch.zeros_like(ray_bundle.origins[..., :1])
    if ray_bundle.fars is None:
        ray_bundle.fars = torch.full_like(ray_bundle.origins[..., :1], fill_value=1000.)
    bbox = mi.BoundingBox3f(mi.Point3f(-bound_r), mi.Point3f(bound_r))
    o = dr.unravel(mi.Point3f, mi.TensorXf(ray_bundle.origins).array)
    v = dr.unravel(mi.Vector3f, mi.TensorXf(ray_bundle.directions).array)
    ray = mi.Ray3f(o, v)
    inter_mask, sol_l, sol_h = bbox.ray_intersect(ray)
    mask_intersect = inter_mask.torch().view(*ray_bundle.nears.shape[:-1]).bool()
    far = sol_h.torch().view(*ray_bundle.fars.shape[:-1], -1)
    ray_bundle.nears[mask_intersect] = far[mask_intersect]


def light_pc_from_rgb_depth(origins, directions, lum, depth):
    pos_samples = origins + directions * depth
    dir_samples = -directions
    lum_samples = lum
    return {
        'pos_samples': pos_samples.view(-1, pos_samples.shape[-1]),
        'dir_samples': dir_samples.view(-1, dir_samples.shape[-1]),
        'lum_samples': lum_samples.view(-1, lum_samples.shape[-1]),
    }


def extract_light_point_cloud(
        pipeline: Pipeline,
        torch_mi2gl_left: torch.Tensor,
        scene_scale: float = 1.0,
        output_filename=None,
        ray_source: Literal["spherical", "training"] = 'training',
        adjoint_sampling_strategy='adjoint',
        camera_scaling_factor=0.25,
        crop_bbox: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        torch_mi2gl_left: transform mitsuba space to nerfstudio space
        scene_scale: ignore inner sphere centered at origin, with radius bound_r
        output_filename: Name of the output file.
        ray_source: Whether we use training images to obtain ray proposal
        adjoint_sampling_strategy: path guiding strategy
        camera_scaling_factor: How much to downsample when choosing ray_source == 'training'
        crop_bbox: Whether it crops the object bbox when rendering light images
    """
    CONSOLE.print("[bold green]Outputting Lighting PointCloud")
    if output_filename is not None:
        os.makedirs(output_filename, exist_ok=True)
    rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=pipeline.device, dtype=torch.float32)

    if ray_source == 'spherical':
        camera_idx = 0
        if pipeline.rotater is not None:
            # Always use the non-rotated space, thus no need to consider a different bounding box
            camera_idx = pipeline.rotater.map_rotation_option_to_camera_idx('0.0')
        camera_ray_bundle = generate_rays_spherical(4096, 2048, radius=0., inward=False, device=pipeline.device,
                                                    camera_idx=camera_idx)
        update_bbox_near_far(camera_ray_bundle, scene_scale)
    elif ray_source == 'training':
        # generate each ray of each pixel in each training image
        cameras = pipeline.datamanager.train_dataset.cameras.to(pipeline.device)
        cameras.rescale_output_resolution(camera_scaling_factor)
        camera_indices = torch.arange(len(cameras), device=pipeline.device)[..., None]
        if crop_bbox:
            bounding_box_min = torch.tensor([-scene_scale, -scene_scale, -scene_scale], dtype=torch.float32,
                                            device=pipeline.device)
            bounding_box_max = torch.tensor([scene_scale, scene_scale, scene_scale], dtype=torch.float32,
                                            device=pipeline.device)
            aabb_box = SceneBox(
                torch.stack([bounding_box_min, bounding_box_max]).to(pipeline.device),
                CropMode.FAR2INF,
            )
            if pipeline.rotater is not None:
                pipeline.rotater.apply_scene_box(aabb_box, camera_indices)
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_indices, aabb_box=aabb_box)
        else:
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_indices)
        if pipeline.rotater is not None:
            camera_ray_bundle.rotater = pipeline.rotater.apply_frustums
    else:
        raise ValueError(f'unknown {ray_source}')

    outputs = pipeline.model.get_point_lights_for_camera_ray_bundle(camera_ray_bundle)
    luminance = (outputs["rgb"] * rgb_weights).sum(dim=-1, keepdim=True).reshape(*camera_ray_bundle.shape, -1)

    def pos_mi2gl(o):
        o = o[None, ...]
        v = o.clone()
        ray_bundle = get_ray_bundle(o, v, torch_mi2gl_left, scene_scale, 0)
        return ray_bundle.origins[0]

    jac_mi2gl = torch.func.jacfwd(pos_mi2gl)(torch_mi2gl_left.new_ones(3))
    brightness_grad_mi = outputs["brightness_grad"] @ jac_mi2gl
    brightness_grad_mi = brightness_grad_mi.reshape(*camera_ray_bundle.shape, -1)

    gradient_norms = []
    if adjoint_sampling_strategy in ['adjoint', 'primal_adjoint']:
        gradient_norms = [brightness_grad_mi.abs().mean(dim=-1, keepdim=True)]

    depth = outputs['depth'].reshape(*camera_ray_bundle.shape, -1)

    primal_outputs = light_pc_from_rgb_depth(
        camera_ray_bundle.origins,
        camera_ray_bundle.directions,
        luminance,
        depth)
    adjoint_outputs = [light_pc_from_rgb_depth(
        camera_ray_bundle.origins,
        camera_ray_bundle.directions,
        gradient_norm,
        depth) for gradient_norm in gradient_norms]

    if output_filename is not None:
        rgb_samples = outputs['rgb'].reshape(camera_ray_bundle.size, -1)
        output_mesh = trimesh.PointCloud(vertices=primal_outputs['pos_samples'].cpu().numpy(),
                                         colors=linear_to_srgb(rgb_samples.cpu().numpy()))
        output_mesh.export(os.path.join(output_filename, f'pc_primal.ply'))
        for i in range(len(adjoint_outputs)):
            output_mesh = trimesh.PointCloud(vertices=adjoint_outputs[i]['pos_samples'].cpu().numpy(),
                                             colors=linear_to_srgb(rgb_samples.cpu().numpy()))
            output_mesh.export(os.path.join(output_filename, f'pc_adjoint_{i}.ply'))

    return primal_outputs, list(adjoint_outputs)


def compensate_pc(lum_samples, pos_samples, dir_samples, output_filename=None, threshold=None):
    lum_samples_compensated = torch.clamp(lum_samples - lum_samples.mean(), 0.)
    mask = lum_samples_compensated[..., 0] > 0
    if threshold is not None:
        mask &= lum_samples[..., 0] > threshold
    pos_samples_masked = pos_samples[mask]
    lum_samples_masked = lum_samples_compensated[mask]
    res = {
        'position': pos_samples_masked,
        'weight': lum_samples_masked,
    }
    if output_filename is not None:
        os.makedirs(output_filename, exist_ok=True)
        output_mesh = trimesh.PointCloud(vertices=pos_samples_masked.cpu().numpy())
        output_mesh.export(os.path.join(output_filename, f'pc_compensated.ply'))
    return res
