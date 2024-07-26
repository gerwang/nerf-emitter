import os
from typing import Dict, Any

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

from emitters.nerf_op import torch_to_drjit
from emitters.util import affine_left
from nerfstudio.model_components.gmm_cluster_light import pc_to_spheres, cluster_light
from nerfstudio.model_components.output_light_pc import compensate_pc, extract_light_point_cloud
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.mi_gl_conversion import torch_point_scale_shifted_gl2mi
from nerfstudio.utils.rich_utils import CONSOLE
from .path_guiding import PathGuiding

N_CLUSTER = 64


def build_spherical_gaussian_clusters(pipeline, adjoint_sampling_strategy):
    primal_light_pc, adjoint_light_pcs = extract_light_point_cloud(
        pipeline,
        torch_mi2gl_left=pipeline.torch_mi2gl_left,
        scene_scale=pipeline.scene_scale,
        ray_source=pipeline.config.ray_source,
        adjoint_sampling_strategy=adjoint_sampling_strategy,
        crop_bbox=pipeline.config.crop_bbox,
    )
    primal_light_pc = compensate_pc(**primal_light_pc, threshold=pipeline.config.primal_threshold)
    for i in range(len(adjoint_light_pcs)):
        adjoint_light_pcs[i] = compensate_pc(**adjoint_light_pcs[i], threshold=pipeline.config.adjoint_threshold)
    primal_light_pc['position'] = torch_point_scale_shifted_gl2mi(
        affine_left(pipeline.torch_gl2mi_left, primal_light_pc['position']),
        pipeline.scene_scale)
    primal_cluster = cluster_light(**primal_light_pc, n_trials=1, n_cluster=N_CLUSTER,
                                   equalize=pipeline.config.equalize, equalize_base=pipeline.config.equalize_base,
                                   ignore_weight=pipeline.config.ignore_weight)
    adjoint_clusters = []
    for i, adjoint_light_pc in enumerate(adjoint_light_pcs):
        adjoint_light_pc['position'] = torch_point_scale_shifted_gl2mi(
            affine_left(pipeline.torch_gl2mi_left, adjoint_light_pc['position']),
            pipeline.scene_scale)
        adjoint_clusters.append(
            cluster_light(**adjoint_light_pc, n_trials=1, n_cluster=N_CLUSTER,
                          equalize=pipeline.config.equalize, equalize_base=pipeline.config.equalize_base,
                          ignore_weight=pipeline.config.ignore_weight))
    torch.cuda.empty_cache()
    return primal_cluster, adjoint_clusters


def load_vmf_params(emitter, cluster):
    vmf_params = mi.traverse(emitter)
    vmf_params['position'] = mi.Point3f(cluster['position'])
    vmf_params['weight'] = mi.Float(cluster['weight'])
    vmf_params['std'] = mi.Float(cluster['std'])
    vmf_params.update()


class VonMisesFisherGuiding(PathGuiding):

    @classmethod
    def emitter_scene_xml(cls, pipeline, asset_path):
        return 'emitters/vmf.xml'

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):

        def dump_vmf_params(emitter, name):
            vmf_params = mi.traverse(emitter)
            for key in ['position', 'weight', 'std']:
                destination[prefix + f'{name}.{key}'] = dr.ravel(vmf_params[key]).torch()

        dump_vmf_params(self.sdf_scene.environment(), 'vmf')
        for i, adjoint_emitter in enumerate(self.sdf_scene.integrator().adjoint_emitters):
            dump_vmf_params(adjoint_emitter, f'adjoint_emitter_{i}')

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int):

        def unravel_or_float(dtype, x):
            if not issubclass(dtype, dr.ArrayBase) or dtype.Depth == 1:
                return dtype(dr.ravel(x))
            else:
                return dr.unravel(dtype, x)

        def load_vmf_params(emitter, name):
            vmf_params = mi.traverse(emitter)
            for key in ['position', 'weight', 'std']:
                if f'{name}.{key}' in loaded_state:
                    vmf_params[key] = unravel_or_float(
                        type(vmf_params[key]),
                        torch_to_drjit(loaded_state[f'{name}.{key}'].to('cuda:0'))
                    )
                    del loaded_state[f'{name}.{key}']
            vmf_params.update()

        load_vmf_params(self.sdf_scene.environment(), 'vmf')
        for i, adjoint_emitter in enumerate(self.sdf_scene.integrator().adjoint_emitters):
            load_vmf_params(adjoint_emitter, f'adjoint_emitter_{i}')

    def output_emitter_proposal(self, output_filename):
        def output_vmf(emitter, name):
            vmf_params = mi.traverse(emitter)
            sphere_mesh = pc_to_spheres(np.asarray(vmf_params['position']), np.asarray(vmf_params['std']),
                                        np.asarray(vmf_params['weight']))
            os.makedirs(f'{output_filename}/{name}', exist_ok=True)
            sphere_mesh.export(f'{output_filename}/{name}/light_sphere.ply')

        output_vmf(self.sdf_scene.environment(), 'primal')
        for i, adjoint_emitter in enumerate(self.sdf_scene.integrator().adjoint_emitters):
            output_vmf(adjoint_emitter, f'adjoint_{i}')
        CONSOLE.log(f'Save emitter proposal to {output_filename}')

    def build_emitter_proposal(self, pipeline: Pipeline):
        primal_cluster, adjoint_clusters = build_spherical_gaussian_clusters(pipeline, self.adjoint_sampling_strategy)

        load_vmf_params(pipeline.sdf_scene.environment(), primal_cluster)
        for i in range(len(adjoint_clusters)):
            load_vmf_params(pipeline.sdf_scene.integrator().adjoint_emitters[i], adjoint_clusters[i])
