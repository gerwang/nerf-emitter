import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import trimesh.creation
from matplotlib.colors import Normalize
from pomegranate.distributions import *
from pomegranate.gmm import GeneralMixtureModel

from nerfstudio.utils.rich_utils import CONSOLE


def pc_to_spheres(position, stdev, weight):
    meshes = []
    weight_colors = weight_to_heat(weight)
    for i in range(position.shape[0]):
        sphere = trimesh.creation.uv_sphere()
        sphere.apply_scale(stdev[i].mean())
        sphere.apply_translation(position[i])
        sphere.visual.vertex_colors = np.tile(weight_colors[i:i + 1], (sphere.vertices.shape[0], 1))
        meshes.append(sphere)
    spheres = trimesh.util.concatenate(meshes)
    return spheres


def weight_to_heat(weight_values):
    # Create a colormap (you can use different colormaps)
    cmap = plt.get_cmap('viridis')  # You can replace 'viridis' with any other colormap

    # Normalize the weight values to the range [0, 1]
    norm = Normalize(vmin=0, vmax=2. / len(weight_values))

    # Convert weight values to colors using vectorized operations
    colors = cmap(norm(weight_values))
    return colors


def cluster_light(position: torch.Tensor, weight: torch.Tensor, n_cluster=64,
                  output_filename=None, n_trials=3, verbose=False, equalize=False,
                  equalize_base=10.0, ignore_weight=False) -> Dict[str, Any]:
    if equalize:
        num_points = torch.ceil(weight / equalize_base)[..., 0].int()
        weight_divided = weight / num_points[..., None]
        position = position.repeat_interleave(num_points, dim=0)
        weight = weight_divided.repeat_interleave(num_points, dim=0)
    if ignore_weight:
        weight[:] = 1.0
    best_logp = None
    position_gmm = None
    std_gmm = None
    weight_gmm = None
    for i in range(n_trials):
        while True:
            try:
                position_noisy = position + torch.randn_like(position) * 1e-6
                model = GeneralMixtureModel(
                    [Normal(covariance_type='sphere') for _ in range(n_cluster)],
                    verbose=verbose
                ).to(position.device).fit(position_noisy, sample_weight=weight)
                logp = model.summarize(position_noisy, sample_weight=weight)
                if verbose:
                    CONSOLE.log(f'Trial {i}, logp {float(logp)}')
                if best_logp is None or best_logp < logp:
                    best_logp = float(logp)
                    position_gmm = np.stack([x.means.cpu().numpy() for x in model.distributions])
                    std_gmm = np.stack([np.sqrt(x.covs.cpu().numpy()) for x in model.distributions])
                    weight_gmm = model.priors.cpu().numpy()
                del model
                del logp
                break
            except Exception as e:
                CONSOLE.log(f'[bold red]Encounter exception: "{str(e)}", retrying')

    if output_filename is not None:
        os.makedirs(output_filename, exist_ok=True)
        sphere_mesh = pc_to_spheres(position_gmm, std_gmm, weight_gmm)
        sphere_mesh.export(f'{output_filename}/light_sphere.ply')
    return {
        'position': position_gmm,
        'std': std_gmm,
        'weight': weight_gmm,
    }
