# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A pipeline that optimize both mitsuba sdf and nerf since a few iterations.
"""
from __future__ import annotations

import functools
import os
import pathlib
from collections import OrderedDict
from dataclasses import dataclass, field
from time import time
from typing import Literal, Type, Optional, Dict, Any, List, cast

import drjit as dr
import imageio.v3 as iio
import mitsuba as mi
import numpy as np
import torch
from jaxtyping import Float
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from torch import Tensor
from torch.autograd import forward_ad
from torch.cuda.amp.grad_scaler import GradScaler

import losses
import redistancing
import regularizations as reg
from configs import get_config
from constants import SDF_DEFAULT_KEY_P, ENV_DEFAULT_KEY, NERF_DEFAULT_KEY
from emitters import register_emitters
from emitters.env_emitter_op import env_emitter_op
from emitters.nerf_emitter_op import nerf_emitter_op
from emitters.nerf_op import torch_to_drjit, get_ray_bundle, pad_gather, pad_scatter, scatter_camera_idx
from fd_util import eval_forward_gradient, mi_create_cube_sdf
from integrators import import_integrators
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import CropMode
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.exporter.tsdf_utils import TSDF
from nerfstudio.model_components.mi_sensor_generators import MitsubaSensorGenerator
from nerfstudio.path_guiding import get_path_guiding_class
from nerfstudio.path_guiding.path_guiding import PathGuiding
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler, comms
from nerfstudio.utils.colormaps import ColormapOptions, linear_to_srgb_torch
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils.mi_gl_conversion import mi2gl_left, gl2mi_left, torch_scale_shifted_gl2mi
from nerfstudio.utils.mi_util import render_aggregate, divide_spp, clear_memory, disable_aabb
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.render_utils import render_trajectory_video, CropData, indices_by_rank
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import GLOBAL_BUFFER
from opt_configs import get_opt_config
from util import set_sensor_res, atleast_4d
from variables import VolumeVariable, SdfVariable


@dataclass
class MitsubaSdfPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: MitsubaSdfPipeline)
    takeover_step: int = 30000
    """Step to use Mitsuba SDF"""
    mi_opt_config_name: str = 'principled-12-relativel1-hqq'
    """Optimization configurations to run"""
    mi_config_name: str = 'warponemis'
    """Method to be used for the optimization. Default: warponemis"""
    llvm: bool = False
    """Force use of LLVM (CPU) mode instead of CUDA/OptiX. This can be useful if compilation times using OptiX are too long."""
    verbose: bool = False
    """Print additional log information"""
    load_mean_step: int = 30511
    """The step to load mean parameters"""
    light_pc_cnt: int = 32768
    """Only preserve this amount of light point cloud when building emitter proposal"""
    override_trainer_config: bool = False
    """Whether it overrides trainer's configs at takeover_step"""
    no_update_nerf: bool = False
    """Do no update nerf when joint optimization"""
    spp_per_batch: int = 256
    """Number of spp per batch, to avoid MLE"""
    detach_op: bool = False
    """Whether it detaches the nerf emitter op"""
    emitter_once: bool = False
    """Only build emitter proposal at the start of the inverse rendering"""
    render_internal_mask: bool = False
    """Export object mask rendered by NeRF to improve shape convergence"""
    use_internal_mask: bool = False
    """Use object mask rendered by NeRF to improve shape convergence"""
    mask_loss_mult: float = 10.0
    """Mask loss multiplier."""
    view_loss_mult: float = 1.0
    """View loss multiplier."""
    hide_emitters: bool = False
    """Whether hide emitters when Mitsuba rendering"""
    tsdf_batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_tsdf_init: bool = True
    """Whether uses tsdf init (if using mask loss)"""
    save_mask: bool = False
    """Whether output masks"""
    save_envmap: bool = False
    """Whether output environment maps"""
    envmap_last_save_exr: bool = True
    """Save last iter envmap as exr"""
    mi_loss_type: Optional[str] = None
    """Render loss type"""
    mi_learning_rate: Optional[float] = None
    """Mitsuba SDF learning rate"""
    texture_lr: Optional[float] = None
    """Mitsuba texture learning rate"""
    laplacian_loss_mult: Optional[float] = None
    """Laplacian loss multiplier"""
    texture_laplacian_loss_mult: Optional[float] = None
    """Texture Laplacian loss multiplier"""
    redistance_freq: Optional[int] = None
    """Redistance frequency"""
    mask_optimizer: bool = True
    """Whether it uses SparseAdam"""
    voxel_upsample_iter: Optional[List[int]] = None
    """Voxel upsample iter"""
    image_upsample_iter: Optional[List[int]] = None
    """Image upsample iter"""
    mi_res_x: Optional[int] = None
    """Mitsuba resolution x"""
    mi_res_y: Optional[int] = None
    """Mitsuba resolution y"""
    mi_batch_size: Optional[int] = None
    """How many images to render in an iteration"""
    sdf_res: Optional[int] = None
    """Mitsuba SDF resolution"""
    save_voxel_path: Optional[pathlib.Path] = None
    """Save voxel to the specific path"""
    save_voxel_inplace: bool = True
    """Save TSDF inited SDF voxel at output dir"""
    load_voxel_path: Optional[pathlib.Path] = None
    """Load voxel at specific path"""
    adaptive_learning_rate: Optional[bool] = None
    """Mitsuba use LR decay"""
    tsdf_res: int = 256
    """TSDF voxel resolution when init"""
    spp: Optional[int] = None
    """Number of spp, to avoid MLE"""
    mesh_xml_path: Optional[pathlib.Path] = None
    """Mesh XML Path, if enabled, will not use SDF but use mesh instead"""
    emitter_xml_path: Optional[pathlib.Path] = None
    """Override emitter scene path"""
    sdf_cube_size: Optional[float] = None
    """Override sdf voxel with a cube"""
    roughness_override: Optional[float] = None
    """Override roughness"""
    albedo_override: Optional[List[float]] = None
    """Override albedo"""
    primal_spp_mult: Optional[int] = None
    """Override primal spp multiplier"""
    adjoint_sampling_strategy: Literal['primal'] = 'primal'
    """What emitter sampling strategy we use in adjoint rendering?"""
    guiding_type: Literal['vmf', 'env', 'emitter_xml'] = 'vmf'
    """Path guiding type"""
    guiding_mis_compensation: bool = True
    """Whether it enables mis compensation when training path guiding"""
    query_emitter_index: Optional[int] = None
    """The emitter index to visualize, -1 means scene.environment()"""
    sdf_position: Optional[List[float]] = None
    """Specify sdf position"""
    render_side_each_iter: bool = False
    """Render a side image every iteration"""
    padding_size: Optional[float] = None
    """Bounding box padding size"""
    curvature_spp: int = 2
    """Number of samples per pixel when computing curvature loss"""
    curvature_loss_mult: float = 0.05
    """Curvature loss multiplier"""
    curvature_epsilon: float = 1e-3
    """Radius when selecting neighbors"""
    power_of_two: bool = False
    """Whether it performs M = ⌊log2(N + 1)⌋ iterations with power-of-two sample counts. 
    This approach was initially proposed by Müller et al. [2017] to limit the impact 
    of initial high-variance estimates on the final image."""
    use_visibility: Optional[bool] = None
    """Whether it uses visibility in direct integrator"""
    render_occlusion: bool = False
    """Whether it renders the RGBA images depicting occlusion in front of cameras"""
    use_occlusion_image: bool = False
    """Load occlusion image and composite it during Mitsuba rendering"""
    bbox_constraint: bool = True
    """Whether to constraint sdf inside the bbox"""
    occlusion_load_dir: Optional[pathlib.Path] = None
    """Which directory to load occlusion and background images"""
    envmap_lr: Optional[float] = None
    """Set environment map learning rate"""
    rough_init_value: Optional[float] = None
    """Override the roughness initial value in Mitsuba"""
    use_nerf_render: bool = False
    """Use NeRF's render_camera_outputs instead of Mitsuba"""
    disable_build_emitter_proposal: bool = False
    """Disable build emitter proposal"""
    crop_bbox: bool = True
    """Whether it crops the object bbox when rendering light images"""
    primal_threshold: float = 1.0
    """Ignore point clouds whose brighness is lower than the threshold"""
    adjoint_threshold: float = 10.0
    """Ignore point aclouds whose gradient is lower than the threshold"""
    ray_source: Literal["spherical", "training"] = 'training'
    """Whether we use training images to obtain ray proposal"""
    equalize: bool = False
    """Equalize point cloud to ease GMM clustering"""
    equalize_base: float = 10.0
    """Minimum equalize weight"""
    ignore_weight: bool = False
    """Ignore luminance weight in importance sampling"""
    steps_per_build_proposal: int = 10
    """How often should the emitter proposal being built"""
    override_eval_images_metrics: bool = False
    """override the default get_average_eval_image_metrics"""
    mesh_path: Optional[pathlib.Path] = None
    """mesh directory path passed to mitsuba xml to load mesh"""
    mesh_type: Literal['diffuse', 'principled'] = 'principled'
    """mesh type in mesh_path"""
    denoise: bool = False
    """Use Optix denoiser when render_camera_outputs"""
    apply_rotation_camera_file: bool = False
    """Apply rotater rotations even if using a camera file"""


class MitsubaSdfPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: MitsubaSdfPipelineConfig

    def takeover_backward(self, step):
        return step >= self.config.takeover_step

    def set_trainer(self, trainer: 'Trainer'):
        super().set_trainer(trainer)

    def __init__(
            self,
            config: MitsubaSdfPipelineConfig,
            device: str,
            test_mode: Literal["test", "val", "inference"] = "val",
            world_size: int = 1,
            local_rank: int = 0,
            grad_scaler: Optional[GradScaler] = None,
            mixed_precision: bool = False,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler, mixed_precision)
        self.torch_mi2gl_left = torch.from_numpy(mi2gl_left).float().to(self.device)
        self.torch_gl2mi_left = torch.from_numpy(gl2mi_left).float().to(self.device)
        self.scene_scale = self.datamanager.train_dataparser_outputs.dataparser_scale
        self.rotater = getattr(self.datamanager, 'rotater', None)

        self.mi_config = get_config(self.config.mi_config_name)

        def filter_none(d):
            res = {}
            for k, v in d.items():
                if isinstance(v, tuple):
                    f, v = v
                else:
                    f = None
                if v is not None:
                    res[k] = f(v) if f is not None else v
            return res

        self.opt_config, unused_args = get_opt_config(self.config.mi_opt_config_name, cmd_args=filter_none({
            'sdf_res': self.config.sdf_res,
            'upsample_iter': self.config.voxel_upsample_iter,
            'tex_upsample_iter': self.config.voxel_upsample_iter,
            'rough_upsample_iter': self.config.voxel_upsample_iter,
            'learning_rate': self.config.mi_learning_rate,
            'texture_lr': self.config.texture_lr,
            'loss': (lambda x: getattr(losses, x), self.config.mi_loss_type),
            'resx': self.config.mi_res_x,
            'resy': self.config.mi_res_y,
            'render_upsample_iter': self.config.image_upsample_iter,
            'use_multiscale_rendering': (lambda x: str(len(x) > 0), self.config.image_upsample_iter),
            'batch_size': self.config.mi_batch_size,
            'adaptive_learning_rate': (lambda x: str(x), self.config.adaptive_learning_rate),
            'bbox_constraint': (lambda x: str(x), self.config.bbox_constraint),
            'envmap_lr': self.config.envmap_lr,
            'rough_init_value': self.config.rough_init_value,
        }))
        assert len(unused_args) == 0

        # apply optimization configs
        if self.config.spp is not None:
            self.mi_config.spp = self.config.spp
        else:
            self.config.spp = self.mi_config.spp
        if self.config.primal_spp_mult is not None:
            self.mi_config.primal_spp_mult = self.config.primal_spp_mult
        else:
            self.config.primal_spp_mult = self.mi_config.primal_spp_mult
        self.config.mask_optimizer = self.config.mask_optimizer
        for variable in self.opt_config.variables:
            if isinstance(variable, SdfVariable):
                variable.regularizer = functools.partial(reg.eval_discrete_laplacian_reg,
                                                         sparse=self.config.mask_optimizer)
                if self.config.laplacian_loss_mult is not None:
                    variable.regularizer_weight = self.config.laplacian_loss_mult
            elif isinstance(variable, VolumeVariable):
                variable.regularizer = functools.partial(reg.eval_discrete_laplacian_reg,
                                                         sparse=self.config.mask_optimizer)
                if self.config.texture_laplacian_loss_mult is not None:
                    variable.regularizer_weight = self.config.texture_laplacian_loss_mult
            if hasattr(variable, 'redistance_freq') and self.config.redistance_freq is not None:
                setattr(variable, 'redistance_freq', self.config.redistance_freq)

        self._ddp_cd = 0
        self._step = -1

        import sensors.spherical_sensor  # noqa
        # other processes also need to know mitsuba sensors
        self.init_mitsuba()

    @check_main_thread
    def init_mitsuba(self):
        import_integrators()
        register_emitters()

        use_llvm = self.config.llvm or not ('cuda_ad_rgb' in mi.variants())
        mi.set_variant('llvm_ad_rgb' if use_llvm else 'cuda_ad_rgb')

        ASSET_PATH = './differentiable-sdf-rendering/assets'

        if self.config.emitter_xml_path is not None:
            CONSOLE.log('Overriding guiding_type to emitter_xml')
            self.config.guiding_type = 'emitter_xml'  # for now, always change to an envmap emitter

        path_guiding_type = get_path_guiding_class(self.config.guiding_type)

        mts_args = {
            'hide_emitters': self.config.hide_emitters,
            'guiding_mis_compensation': self.config.guiding_mis_compensation,
            'use_bbox_sdf': not self.config.bbox_constraint,
        }
        mts_args.update(path_guiding_type.mts_args(self, ASSET_PATH))
        if self.config.query_emitter_index is not None:
            mts_args.update({
                'query_emitter_index': self.config.query_emitter_index,
            })
        if self.config.padding_size is not None:
            mts_args.update({
                'padding_size': self.config.padding_size,
            })
        if self.config.mesh_xml_path is not None:
            mts_args.update({
                'extra_mesh_file': os.path.relpath(self.config.mesh_xml_path, ASSET_PATH),
            })
        if self.config.mesh_path is not None:
            mts_args.update({
                'mesh_path': self.config.mesh_path,
                'extra_mesh_file': os.path.relpath(
                    {
                        'diffuse': 'differentiable-sdf-rendering/assets/objects/diffuse_mesh.xml',
                        'principled': 'differentiable-sdf-rendering/assets/objects/principled_mesh.xml',
                    }[self.config.mesh_type], ASSET_PATH),
            })

        self.sdf_scene = mi.load_file(f'{ASSET_PATH}/sdf_scene.xml',
                                      integrator=self.mi_config.integrator,
                                      parallel=self.mi_config.use_parallel_loading,
                                      main_bsdf_name=self.opt_config.main_bsdf_name,
                                      **mts_args)
        self.sdf_object = self.sdf_scene.integrator().sdf
        if self.config.mesh_xml_path is not None or self.config.mesh_path is not None:
            self.sdf_backup_object = self.sdf_object  # keep a reference to the object so that it is not destructed
            self.sdf_object = self.sdf_scene.integrator().sdf = None
        else:
            self.sdf_scene.integrator().warp_field = self.mi_config.get_warpfield(self.sdf_object)
            assert any('_sdf_' in shape.id() for shape in
                       self.sdf_scene.shapes()), "Could not find a placeholder shape for the SDF"
        if self.config.use_visibility is not None and hasattr(self.sdf_scene.integrator(), 'use_visibility'):
            self.sdf_scene.integrator().use_visibility = self.config.use_visibility
        self.path_guiding: PathGuiding = path_guiding_type(self.sdf_scene, self.mi_config,
                                                           self.config.adjoint_sampling_strategy, ASSET_PATH)

        self.params = mi.traverse(self.sdf_scene)
        if self.config.sdf_position is not None:
            self.params[SDF_DEFAULT_KEY_P] = mi.Vector3f(*self.config.sdf_position)
            self.params.update()
        self.params.keep(self.opt_config.param_keys + [NERF_DEFAULT_KEY])

        self.opt = mi.ad.Adam(lr=self.opt_config.learning_rate, params=self.params,
                              mask_updates=self.opt_config.mask_optimizer)
        self.opt_config.initialize(self.opt, self.sdf_scene)
        if NERF_DEFAULT_KEY in self.opt:
            dr.enable_grad(self.opt[NERF_DEFAULT_KEY])
        if SDF_DEFAULT_KEY_P in self.opt and self.config.sdf_position is not None:
            self.opt[SDF_DEFAULT_KEY_P] = mi.Vector3f(*self.config.sdf_position)
        self.params.update(self.opt_config.get_param_dict(self.opt))
        self.register_buffer('seed', torch.tensor(0, dtype=torch.int64))

        if hasattr(self.sdf_scene.environment(), 'set_op'):
            self.sdf_scene.environment().set_op(
                functools.partial(nerf_emitter_op,
                                  pipeline=self,
                                  torch_mi2gl_left=self.torch_mi2gl_left,
                                  scene_scale=self.scene_scale,
                                  path_guiding=self.path_guiding,
                                  rotater=self.rotater)
            )
            self.sdf_scene.environment().detach_op = self.config.detach_op

        # Set initial rendering resolution
        for sensor in self.opt_config.sensors:
            set_sensor_res(sensor, self.opt_config.init_res)

        if self.config.load_voxel_path is not None:
            self.load_voxels(self.config.load_voxel_path)

        self.curvature_integrator = mi.load_file(
            f'{ASSET_PATH}/common.xml',
            integrator='sdf_curvature',
            parallel=self.mi_config.use_parallel_loading,
            curvature_epsilon=self.config.curvature_epsilon,
        ).integrator()
        self.curvature_integrator.sdf = self.sdf_object

        self.depth_integrator = mi.load_file(
            f'{ASSET_PATH}/common.xml',
            integrator='sdf_normal_depth',
            parallel=self.mi_config.use_parallel_loading,
            curvature_epsilon=self.config.curvature_epsilon,
        ).integrator()
        self.depth_integrator.sdf = self.sdf_object

        # torch.cuda.set_per_process_memory_fraction(0.8, self.device)

    def load_voxels(self, load_voxel_path, adaptive_resolution=True):
        for variable in self.opt_config.variables:
            if isinstance(variable, VolumeVariable):
                var_path = variable.get_variable_path(str(load_voxel_path), suffix='')
                if not pathlib.Path(var_path).exists():
                    CONSOLE.log(f'Cannot find load voxel path {var_path}')
                else:
                    data = np.array(mi.VolumeGrid(var_path))
                    variable.load_data(self.opt, data, adaptive_resolution)
        self.params.update(self.opt_config.get_param_dict(self.opt))

    def mi_step(self, step):
        return step - self.config.takeover_step

    def primal_target_size(self):
        film = self.datamanager.mi_train_sensor_generator.film
        spp = self.mi_config.spp * self.mi_config.primal_spp_mult
        return self.get_target_size(film, spp)

    def backward_target_size(self, half=False, spp=None):
        if spp is None:
            spp = self.mi_config.spp
        film = self.datamanager.mi_train_sensor_generator.film
        # if half:
        #     spp //= 2
        return self.get_target_size(film, spp)

    @staticmethod
    def get_target_size(film, spp):
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()
        return int(dr.prod(film_size) * spp)

    def rescale_train_dataset(self, step):
        i = self.mi_step(step)
        if i < 0:
            return
        if i == 0 or self.opt_config.render_upsample_iter is None:
            target_res = self.opt_config.init_res
        else:
            def count_items_not_greater(items, query):
                return sum(item <= query for item in items)

            target_res = self.opt_config.init_res * \
                         2 ** count_items_not_greater(self.opt_config.render_upsample_iter, i)
        mi_train_dataparser_outputs = self.datamanager.mi_train_dataparser_outputs
        original_width = int(mi_train_dataparser_outputs.cameras[0].width)
        target_scale = target_res[0] / original_width
        got_mask = self.config.use_internal_mask and mi_train_dataparser_outputs.mask_filenames is None
        got_occlusion = self.config.use_occlusion_image and (
                mi_train_dataparser_outputs.metadata.get('occlusion_filenames', None) is None or
                mi_train_dataparser_outputs.metadata.get('background_filenames', None) is None)
        if target_scale != self.datamanager.mi_train_dataset.scale_factor or got_mask or got_occlusion:
            if getattr(self, 'trainer', None) is not None:
                if self.config.use_internal_mask:
                    mi_train_dataparser_outputs.mask_filenames = [
                        self.trainer.base_dir / 'internal_masks' / f'{x.stem}.png'
                        for x in mi_train_dataparser_outputs.image_filenames
                    ]
                if self.config.use_occlusion_image:
                    occlusion_load_dir = self.config.occlusion_load_dir
                    if occlusion_load_dir is None:
                        occlusion_load_dir = self.trainer.base_dir
                    mi_train_dataparser_outputs.metadata['occlusion_filenames'] = [
                        occlusion_load_dir / 'occlusion_images' / f'{x.stem}.exr'
                        for x in mi_train_dataparser_outputs.image_filenames
                    ]
                    mi_train_dataparser_outputs.metadata['background_filenames'] = [
                        occlusion_load_dir / 'background_images' / f'{x.stem}.exr'
                        for x in mi_train_dataparser_outputs.image_filenames
                    ]
            self.datamanager.rescale_train(target_scale)
        if target_res[0] >= 512:
            # self.mi_config.primal_spp_mult = self.config.primal_spp_mult // 2
            self.mi_config.spp = self.config.spp // 2
        else:
            # self.mi_config.primal_spp_mult = self.config.primal_spp_mult
            self.mi_config.spp = self.config.spp

    def load_occlusion(self, base_dir):
        mi_train_dataparser_outputs = self.datamanager.mi_train_dataparser_outputs
        target_scale = 1.0
        got_occlusion = self.config.use_occlusion_image and (
                mi_train_dataparser_outputs.metadata.get('occlusion_filenames', None) is None or
                mi_train_dataparser_outputs.metadata.get('background_filenames', None) is None)
        if got_occlusion:
            occlusion_load_dir = self.config.occlusion_load_dir
            if occlusion_load_dir is None:
                occlusion_load_dir = base_dir
            mi_train_dataparser_outputs.metadata['occlusion_filenames'] = [
                occlusion_load_dir / 'occlusion_images' / f'{x.stem}.exr'
                for x in mi_train_dataparser_outputs.image_filenames
            ]
            mi_train_dataparser_outputs.metadata['background_filenames'] = [
                occlusion_load_dir / 'background_images' / f'{x.stem}.exr'
                for x in mi_train_dataparser_outputs.image_filenames
            ]
            self.datamanager.rescale_train(target_scale)

    @check_main_thread
    def load_mean_parameters(self):
        self.opt_config.load_mean_parameters(self.opt)
        self.params.update(self.opt_config.get_param_dict(self.opt))

    @profiler.time_function
    @check_main_thread
    def build_emitter_proposal(self):
        self.path_guiding.build_emitter_proposal(self)
        if getattr(self, 'trainer', None) is not None:
            self.output_emitter_proposal(self.trainer.base_dir / 'emitter_proposal')

    @check_main_thread
    def output_emitter_proposal(self, output_filename):
        self.path_guiding.output_emitter_proposal(output_filename)

    @check_main_thread
    def tsdf_init(self, cameras, depth_images):
        aabb = torch.tensor([
            [0, 0, 0],
            [1, 1, 1],
        ], dtype=torch.float32)
        volume_dims = torch.tensor([self.config.tsdf_res for _ in range(3)])
        tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
        tsdf.to(self.device)
        # camera extrinsics and intrinsics
        c2w: Float[Tensor, "N 4 4"] = to4x4(cameras.camera_to_worlds).to(self.device)
        if self.rotater is not None:
            c2w = self.rotater.apply_c2w_homo(c2w, torch.arange(len(cameras), device=c2w.device).unsqueeze(-1))
        K: Float[Tensor, "N 3 3"] = cameras.get_intrinsics_matrices().to(self.device)
        # Transform SDF to the Mitsuba coordinate space
        c2w = torch_scale_shifted_gl2mi(self.torch_gl2mi_left @ c2w, self.scene_scale)
        depth_images = depth_images.permute(0, 3, 1, 2) / self.scene_scale * 0.5  # shape (N, 1, H, W)

        CONSOLE.print("Integrating the TSDF")
        batch_size = self.config.tsdf_batch_size
        for i in range(0, len(c2w), batch_size):
            tsdf.integrate_tsdf(
                c2w[i: i + batch_size],
                K[i: i + batch_size],
                depth_images[i: i + batch_size],
            )

        CONSOLE.print("Computing Mesh")
        mesh = tsdf.get_mesh()
        CONSOLE.print("Saving TSDF Mesh")
        tsdf.export_mesh(mesh, filename=str(self.trainer.base_dir / "tsdf_mesh.ply"))
        # PyTorch's XYZ order to DrJit's ZYX order
        sdf = atleast_4d(mi.TensorXf(redistancing.redistance(tsdf.values.permute(2, 1, 0) / self.config.tsdf_res)))
        save_voxel_path = self.config.save_voxel_path
        if self.config.save_voxel_inplace:
            save_voxel_path = self.trainer.base_dir / 'tsdf_voxel'
        if save_voxel_path is not None:
            save_voxel_path.mkdir(parents=True, exist_ok=True)
            mi.VolumeGrid(np.array(sdf)).write(
                self.opt_config.variables[0].get_variable_path(str(save_voxel_path), suffix=''))
        self.opt_config.variables[0].load_data(self.opt, sdf)
        self.params.update(self.opt_config.get_param_dict(self.opt))
        torch.cuda.empty_cache()

    def render_internal_mask(self):
        if comms.is_main_process():
            CONSOLE.print(f'[bold yellow]Render internal masks[/bold yellow]')
        if self.datamanager.mi_train_dataset.scale_factor != 1.0:
            self.datamanager.rescale_train(1.0)
        cameras = self.datamanager.mi_train_dataset.cameras
        indices = indices_by_rank(len(cameras))
        depth_images = render_trajectory_video(
            pipeline=self,
            cameras=cameras,
            output_filename=self.trainer.base_dir / 'internal_masks',
            rendered_output_names=['accumulation'],
            return_output_names=['depth', 'accumulation'],
            crop_data=CropData(
                scale=torch.full(size=(3,), fill_value=self.scene_scale * 2, dtype=torch.float32),
            ),
            output_format='images',
            image_format='mask_png',
            colormap_options=ColormapOptions(is_mask=True),
            image_names=[x.stem for x in self.datamanager.mi_train_dataparser_outputs.image_filenames],
            main_process_only=False,
            quiet=True,
            target_device=self.device,
            camera_indices=indices,
        )
        torch.cuda.empty_cache()
        if self.config.use_tsdf_init:
            accumulation_images = torch.stack([depth_images[i] for i in range(1, len(depth_images), 2)], dim=0)
            depth_images = torch.stack([depth_images[i] for i in range(0, len(depth_images), 2)], dim=0)
            depth_images[accumulation_images < 0.5] = 1000.
            if comms.get_world_size() > 1:
                depth_images = pad_gather(depth_images, len(cameras), dim=0)
            self.tsdf_init(cameras, depth_images)
        if comms.get_world_size() > 1:
            comms.synchronize()
        if comms.is_main_process():
            CONSOLE.print(f'[bold green]Done internal masks[/bold green]')

    def render_occlusion(self, output_dirname: str, crop_mode: CropMode, cameras=None):
        if comms.is_main_process():
            CONSOLE.print(f'[bold yellow]Render occlusion images {output_dirname} [/bold yellow]')
        if self.datamanager.mi_train_dataset.scale_factor != 1.0:
            self.datamanager.rescale_train(1.0)
        if cameras is None:
            cameras = self.datamanager.mi_train_dataset.cameras
            base_dir = self.trainer.base_dir
            image_names = [x.stem for x in self.datamanager.mi_train_dataparser_outputs.image_filenames]
        else:
            base_dir = pathlib.Path('')
            image_names = None
        indices = indices_by_rank(len(cameras))
        render_trajectory_video(
            pipeline=self,
            cameras=cameras,
            output_filename=base_dir / output_dirname,
            rendered_output_names=['rgb', 'accumulation'],
            crop_data=CropData(
                scale=torch.full(size=(3,), fill_value=self.scene_scale * 2, dtype=torch.float32),
                crop_mode=crop_mode,
            ),
            output_format='images',
            image_format='exr',
            colormap_options=ColormapOptions(is_mask=True),
            image_names=image_names,
            main_process_only=False,
            quiet=True,
            target_device=self.device,
            camera_indices=indices,
            stack_dim='channel',
        )
        torch.cuda.empty_cache()
        if comms.get_world_size() > 1:
            comms.synchronize()
        if comms.is_main_process():
            CONSOLE.print(f'[bold green]Done occlusion images {output_dirname} [/bold green]')

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def check_no_train(step):
            if step >= self.config.takeover_step and self.trainer.optimizers.schedulers is not None:
                self.no_train()

        if self.config.no_update_nerf:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=check_no_train,
                )
            )

        def check_internal_mask(step):
            if step == self.config.takeover_step:
                self.render_internal_mask()

        if self.config.render_internal_mask:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=check_internal_mask,
                )
            )

        def check_render_occlusion(step):
            if step == self.config.takeover_step:
                self.render_occlusion('occlusion_images', CropMode.NEAR2INF)
                self.render_occlusion('background_images', CropMode.FAR)

        if self.config.render_occlusion:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=check_render_occlusion,
                )
            )

        def check_override_trainer_config(step):
            if self.config.override_trainer_config and self.trainer is not None and step >= self.config.takeover_step:
                GLOBAL_BUFFER["steps_per_log"] = self.trainer.config.logging.steps_per_log = 1
                self.trainer.config.steps_per_eval_batch = 5
                self.trainer.config.steps_per_save = 10

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=check_override_trainer_config,
            )
        )

        def check_rescale(step):
            self.rescale_train_dataset(step)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=check_rescale,
            )
        )

        def check_build_proposal(step):  # first rescale, then build proposal
            self._step = step
            if self.should_build_emitter_proposal():
                self.build_emitter_proposal()

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=check_build_proposal,
            )
        )

        def check_load_mean(step):
            if step == self.config.load_mean_step:
                self.load_mean_parameters()

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=check_load_mean,
            )
        )

        if self.config.render_side_each_iter:
            side_sensors = mi.load_file('differentiable-sdf-rendering/assets/sensors/side.xml').sensors()
            side_sensor_idx = 0
            if comms.is_main_process():
                visualize_output_dir = self.trainer.model_output_dir / '..' / 'visualizations'
                if not visualize_output_dir.exists():
                    visualize_output_dir.mkdir(parents=True, exist_ok=True)

            def render_each_iter(step):
                if step >= self.config.takeover_step:
                    outputs = self.render_camera_outputs(side_sensors, side_sensor_idx)
                    if comms.is_main_process():
                        output_image = outputs['rgb']
                        mi.util.write_bitmap(str(visualize_output_dir / f"{step}.png"),
                                             mi.Bitmap(output_image.cpu().numpy()))

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=render_each_iter,
                )
            )

        return callbacks

    def should_build_emitter_proposal(self):
        if self.config.disable_build_emitter_proposal:
            return False
        if not self.training:
            return False
        if self._step < 0:
            return False
        i = self.mi_step(self._step)
        if self.config.no_update_nerf or self.config.emitter_once:
            return i == 0
        else:
            return i >= 0 and i % self.config.steps_per_build_proposal == 0

    def no_train(self):
        for opt in self.trainer.optimizers.optimizers.values():
            for param_group in opt.param_groups:
                param_group['lr'] = 0
                param_group['init_lr'] = 0
        for param_group in self.get_param_groups().values():
            for param in param_group:
                param.requires_grad_(False)
        self.trainer.optimizers.schedulers = {}

    @check_main_thread
    def mi_opt_step(self, step):
        i = self.mi_step(step)
        self.opt_config.validate_gradients(self.opt, i)
        self.opt.step()
        if self.opt_config.validate_params(self.opt, i):
            clear_memory()
        self.opt_config.update_scene(self.sdf_scene, i)
        self.params.update(self.opt_config.get_param_dict(self.opt))

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        if comms.is_main_process():
            for var in self.opt_config.variables:
                param = var.export(self.opt).torch()
                destination[prefix + var.k] = param if keep_vars else param.detach()

            self.path_guiding.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return destination

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        if comms.is_main_process():
            for var in self.opt_config.variables:
                mi_value = None
                if var.k.endswith('sdf.data') and self.config.sdf_cube_size is not None:
                    mi_value = mi_create_cube_sdf(dr.shape(self.opt[var.k]), self.config.sdf_cube_size)
                elif (var.k.endswith('reflectance.volume.data') or var.k.endswith(
                        'base_color.volume.data')) and self.config.albedo_override is not None:
                    mi_value = dr.empty(mi.TensorXf, dr.shape(self.opt[var.k]))
                    for i in range(len(self.config.albedo_override)):
                        mi_value[..., i] = self.config.albedo_override[i]
                elif var.k.endswith('roughness.volume.data') and self.config.roughness_override is not None:
                    mi_value = dr.full(mi.TensorXf, self.config.roughness_override, dr.shape(self.opt[var.k]))
                elif var.k in loaded_state:
                    mi_value = torch_to_drjit(loaded_state[var.k].to(self.device))
                if var.k in loaded_state:
                    del loaded_state[var.k]
                if mi_value is not None:
                    var.load(self.opt, mi_value, self.mi_step(step))

            self.params.update(self.opt_config.get_param_dict(self.opt))

            self.path_guiding.load_pipeline(loaded_state, step)

        super().load_pipeline(loaded_state, step)

    def handle_get_rgb(self, target_size: int):
        recv_payload = pad_scatter((target_size, 6), self.device, 0)
        o, v = torch.split(recv_payload, [3, 3], dim=-1)
        camera_idx = scatter_camera_idx()
        ray_bundle = get_ray_bundle(o, v, self.torch_mi2gl_left, self.scene_scale, camera_idx, self.rotater)
        rgb = self.model.get_rgb_for_camera_ray_bundle(ray_bundle)
        pad_gather(rgb, target_size, 0)
        # outputs = self.model.get_point_lights_for_camera_ray_bundle(ray_bundle)
        # rgb, depth = outputs['rgb'], outputs['depth']
        # payload = torch.cat([rgb, depth], dim=-1)
        # pad_gather(payload, target_size, 0)

    def handle_backward(self, target_size: int):
        recv_payload = pad_scatter((target_size, 9), self.device, 0)
        o, v, grad_out = torch.split(recv_payload, [3, 3, 3], dim=-1)
        o.requires_grad = True
        v.requires_grad = True
        camera_idx = scatter_camera_idx()
        ray_bundle = get_ray_bundle(o, v, self.torch_mi2gl_left, self.scene_scale, camera_idx, self.rotater)
        self.backward_for_camera_ray_bundle(ray_bundle, grad_out)
        grad = torch.cat([o.grad, v.grad], dim=-1)
        pad_gather(grad, target_size, 0)

    def handle_forward(self, target_size: int):
        recv_payload = pad_scatter((target_size, 12), self.device, 0)
        o, v, grad_o, grad_v = torch.split(recv_payload, [3, 3, 3, 3], dim=-1)
        with forward_ad.dual_level():
            o = forward_ad.make_dual(o, grad_o)
            v = forward_ad.make_dual(v, grad_v)
            camera_idx = scatter_camera_idx()
            ray_bundle = get_ray_bundle(o, v, self.torch_mi2gl_left, self.scene_scale, camera_idx, self.rotater)
            ray_bundle.origins, grad_o = forward_ad.unpack_dual(ray_bundle.origins)
            ray_bundle.directions, grad_v = forward_ad.unpack_dual(ray_bundle.directions)
        grad_rgb = self.model.forward_grad_for_camera_ray_bundle(ray_bundle, grad_o, grad_v)
        pad_gather(grad_rgb, target_size, 0)

    @profiler.time_function
    def get_train_loss_dict(self, step: int, model_output_dir: pathlib.Path | None = None):
        if not self.takeover_backward(step):
            return super().get_train_loss_dict(step)
        loss_dict = {}
        model_outputs = []
        self.model.clear_other_losses()
        self.set_ddp_period()
        with disable_aabb(self.model):
            if comms.is_main_process():
                self.path_guiding.set_training_period(self.opt_config.batch_size * (self.mi_config.primal_spp_mult + 1),
                                                      self.opt_config.batch_size)
                for i in range(self.opt_config.batch_size):
                    sensor, batch = self.datamanager.next_train_mitsuba(step)
                    img_index = int(batch['image_idx'])
                    if self.rotater is not None:
                        if comms.is_main_process():
                            self.rotater.apply_sdf_scene(self.sdf_scene, img_index)
                        self.rotater.apply_mi_sensor(sensor, img_index)
                    model_output = {
                        'rgb': 0,
                        'mask': 0,
                        'index': img_index,
                    }
                    seed = self.seed + img_index * self.mi_config.primal_spp_mult
                    seed_grad = self.seed + (
                            img_index + len(self.datamanager.mi_train_dataset)) * self.mi_config.primal_spp_mult
                    img = render_aggregate(
                        self.sdf_scene,
                        spp=self.mi_config.spp * self.mi_config.primal_spp_mult,
                        spp_per_batch=self.mi_config.spp,
                        seed=seed,
                        seed_grad=seed_grad,
                        params=self.params, sensor=sensor,
                        power_of_two=self.config.power_of_two,
                    )
                    img, mask = img[..., :3], img[..., 3]
                    if self.config.use_occlusion_image:
                        occlusion = mi.TensorXf(batch['occlusion'].float().to(self.device))
                        occlusion_mask = mi.TensorXf(batch['occlusion_mask'].float().to(self.device))
                        background = mi.TensorXf(batch['background'].float().to(self.device))
                        img = img + (1 - mask[..., None]) * background
                        img = occlusion + (1 - occlusion_mask) * img
                    model_output['rgb'] += img
                    model_output['mask'] += mask
                    self.seed += len(self.datamanager.mi_train_dataset) * self.mi_config.primal_spp_mult * 2
                    ref_img = mi.TensorXf(batch['image'].reshape(img.shape).float().to(self.device))
                    view_loss = (self.config.view_loss_mult *
                                 self.opt_config.loss(img, ref_img) / self.opt_config.batch_size)
                    loss_dict['view_loss'] = loss_dict.get('view_loss', 0) + view_loss.torch()
                    render_loss = view_loss
                    if 'mask' in batch:
                        ref_mask = mi.TensorXf(batch['mask'].reshape(mask.shape).float().to(self.device))
                        mask_loss = self.config.mask_loss_mult * \
                                    self.opt_config.mask_loss(mask, ref_mask) / self.opt_config.batch_size
                        loss_dict['mask_loss'] = loss_dict.get('mask_loss', 0) + mask_loss.torch()
                        render_loss += mask_loss
                    dr.backward(render_loss)

                    if self.config.curvature_loss_mult > 0:
                        curvature_img = mi.render(
                            self.sdf_scene,
                            spp=self.config.curvature_spp,
                            seed=self.seed + img_index * self.mi_config.primal_spp_mult,
                            seed_grad=self.seed + (
                                    img_index + len(
                                self.datamanager.mi_train_dataset)) * self.mi_config.primal_spp_mult,
                            params=self.params, sensor=sensor,
                            integrator=self.curvature_integrator,
                        )
                        curvature_img, curvature_mask = curvature_img[..., :3], curvature_img[..., 3]
                        curvature_cnt = dr.sum(curvature_mask)
                        curvature_loss = mi.Float(0.)
                        if float(curvature_cnt.torch()) > 0:
                            curvature_loss = self.config.curvature_loss_mult * dr.sum(curvature_img) / curvature_cnt
                        if dr.grad_enabled(curvature_loss):
                            dr.backward(curvature_loss)
                        loss_dict['curvature_loss'] = loss_dict.get('curvature_loss', 0) + curvature_loss.torch()

                    model_outputs.append(model_output)
                    if model_output_dir is not None:
                        if not model_output_dir.exists():
                            model_output_dir.mkdir(parents=True, exist_ok=True)
                        mi.util.write_bitmap(
                            str(model_output_dir / f'{step}-{i}-{img_index}.png'),
                            mi.Bitmap(model_output['rgb'])
                        )
                        if self.config.save_mask:
                            mask_output_dir = model_output_dir.parent / 'mask_outputs'
                            if not mask_output_dir.exists():
                                mask_output_dir.mkdir(parents=True, exist_ok=True)
                            mi.util.write_bitmap(
                                str(mask_output_dir / f'{step}-{i}-{img_index}.png'),
                                mi.Bitmap(model_output['mask'])
                            )
                reg_loss = self.opt_config.eval_regularizer(self.opt, self.sdf_object, step)
                if dr.grad_enabled(reg_loss):
                    dr.backward(reg_loss)
                loss_dict['reg_loss'] = reg_loss.torch()
                self.opt_config.update_loss_dict(self.opt, loss_dict)
            else:
                for i in range(self.opt_config.batch_size):
                    iter_spps = divide_spp(spp=self.mi_config.spp * self.mi_config.primal_spp_mult,
                                           spp_per_batch=self.mi_config.spp,
                                           power_of_two=self.config.power_of_two)
                    for iter_spp in iter_spps:
                        self.handle_get_rgb(self.backward_target_size(spp=iter_spp))
                    last_spp = iter_spps[-1]
                    self.handle_get_rgb(self.backward_target_size(spp=last_spp))
                    if not self.config.detach_op:
                        self.handle_backward(self.backward_target_size(half=False, spp=last_spp))
        # other nerfacto losses
        loss_dict.update(self.model.other_losses)
        if self.config.no_update_nerf:
            loss_dict.update({'dummy_loss': 0.})
        self.model.clear_other_losses()

        if model_output_dir is not None and self.config.save_envmap:
            envmap_output_dir = model_output_dir.parent / 'envmap_outputs'
            if not envmap_output_dir.exists():
                envmap_output_dir.mkdir(parents=True, exist_ok=True)
            envmap = mi.Bitmap(self.params[ENV_DEFAULT_KEY][:, :-1])
            mi.util.write_bitmap(str(envmap_output_dir / f'{step}.png'), envmap)
            if (self.trainer is not None and self.config.envmap_last_save_exr and
                    step + 1 == self.trainer.config.max_num_iterations):
                envmap.write(str(envmap_output_dir / f'{step}.exr'))
        return model_outputs, loss_dict, {}

    def render_camera_outputs(self, cameras, camera_idx, crop_data=None):
        if crop_data is not None or self.config.use_nerf_render:  # output masks
            return super().render_camera_outputs(cameras, camera_idx, crop_data)
        sensor = self.get_mi_sensor(cameras, camera_idx, filter_type='box' if self.config.denoise else 'gaussian')
        outputs = {}
        with disable_aabb(self.model):
            if comms.is_main_process():
                with dr.suspend_grad():
                    kwargs = {}
                    if self.config.denoise:
                        kwargs.update({
                            'denoise_no_gbuffer': True,
                        })
                    img = render_aggregate(self.sdf_scene, sensor=sensor,
                                           spp=self.mi_config.spp * self.mi_config.primal_spp_mult,
                                           spp_per_batch=self.config.spp_per_batch,
                                           power_of_two=self.config.power_of_two, **kwargs)
                    img, mask = img[..., :3], img[..., 3]
                img = img.torch()
                mask = mask.torch().unsqueeze(-1)
                if self.config.mi_config_name == 'warpemitterdistribution':
                    outputs.update({
                        'primal_importance': img[..., 0:1],
                        'adjoint_0_importance': img[..., 1:2],
                        'adjoint_1_importance': img[..., 2:3],
                        'mask': mask,
                    })
                else:
                    outputs.update({
                        'rgb': img,
                        'mask': mask,
                    })
            else:
                iter_spps = divide_spp(spp=self.mi_config.spp * self.mi_config.primal_spp_mult,
                                       spp_per_batch=self.config.spp_per_batch,
                                       power_of_two=self.config.power_of_two)
                for iter_spp in iter_spps:
                    self.handle_get_rgb(self.get_target_size(sensor.film(), iter_spp))
        return outputs

    def render_forward_gradient(self, cameras, camera_idx, axis, spp, spp_per_batch, target_value):
        sensor = self.get_mi_sensor(cameras, camera_idx)
        outputs = {}
        with disable_aabb(self.model):
            if comms.is_main_process():
                img, grad, _ = eval_forward_gradient(self.sdf_scene, self.mi_config, axis=axis, spp=spp,
                                                     fd_spp=spp, sensor=sensor, spp_per_batch=spp_per_batch,
                                                     target_value=target_value, power_of_two=self.config.power_of_two)
                img, mask = img[..., :3], img[..., 3]
                grad, grad_mask = grad[..., :3], grad[..., 3]
                outputs.update({
                    'rgb': img.torch(),
                    'grad': grad.torch(),
                    'mask': mask.torch().unsqueeze(-1),
                    'grad_mask': grad_mask.torch().unsqueeze(-1),
                })
            else:
                iter_spps = divide_spp(spp, spp_per_batch, power_of_two=self.config.power_of_two)
                for iter_spp in iter_spps:
                    forward_target_size = self.get_target_size(sensor.film(), iter_spp)
                    forward_ad_target_size = self.get_target_size(sensor.film(), iter_spp)
                    if self.mi_config.use_finite_differences:
                        self.handle_get_rgb(forward_target_size)
                        self.handle_get_rgb(forward_target_size)
                    else:
                        self.handle_get_rgb(forward_target_size)
                        self.handle_get_rgb(forward_target_size)
                        if not self.config.detach_op:
                            self.handle_forward(forward_ad_target_size)
        return outputs

    def get_mi_sensors(self, cameras, camera_indices, filter_type='gaussian'):
        if isinstance(cameras, list):
            sensors = [cameras[x] for x in camera_indices]
            if self.config.apply_rotation_camera_file:
                for sensor, camera_idx in zip(sensors, camera_indices):
                    # Remember, here we DO apply camera optimizer
                    if self.rotater is not None:
                        self.rotater.apply_mi_sensor(sensor, camera_idx)
                if self.rotater is not None and len(camera_indices) > 0:
                    if comms.is_main_process():
                        self.rotater.apply_sdf_scene(self.sdf_scene, camera_indices[0])
        else:
            mi_sensor_generator = MitsubaSensorGenerator(cameras.to(self.device), self.scene_scale,
                                                         self.datamanager.train_camera_optimizer,
                                                         filter_type=filter_type)
            sensors = []
            for camera_idx in camera_indices:
                sensor = mi_sensor_generator(camera_idx)
                # Remember, here we DO apply camera optimizer
                if self.rotater is not None:
                    self.rotater.apply_mi_sensor(sensor, camera_idx)
                sensors.append(sensor)
            if self.rotater is not None and len(camera_indices) > 0:
                if comms.is_main_process():
                    self.rotater.apply_sdf_scene(self.sdf_scene, camera_indices[0])
                    # Limitation: assume all selected camera_indices have the same rotation state
        return sensors

    def get_mi_sensor(self, cameras, camera_idx, filter_type='gaussian'):
        return self.get_mi_sensors(cameras, [camera_idx], filter_type=filter_type)[0]

    @profiler.time_function
    def backward_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, grad_out: torch.Tensor) -> None:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            grad_out: gradients of the output rgb
        """
        # recover random generator states
        for (g, state) in zip(self.model.get_generators(), self.model.generator_states):
            g.set_state(state)
        num_rays_per_chunk = self.model.config.backward_num_rays_per_chunk
        num_rays = len(camera_ray_bundle)
        self.model._num_processed_rays += num_rays
        cpu_or_cuda_str: str = str(self.device).split(":")[0]

        origins = camera_ray_bundle.origins
        camera_ray_bundle.origins = origins.detach()
        camera_ray_bundle.origins.requires_grad = origins.requires_grad
        directions = camera_ray_bundle.directions
        camera_ray_bundle.directions = directions.detach()
        camera_ray_bundle.directions.requires_grad = directions.requires_grad
        dummy_optimizer = torch.optim.Adam([camera_ray_bundle.origins, camera_ray_bundle.directions])

        # Dummy optimizer used to unscale the gradients of the inputs

        def backward_iter(__ray_bundle, __grad_out_chunk):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.model.mixed_precision):
                outputs, reg_loss_dict = self._model(ray_bundle=__ray_bundle, is_backward=True)
                __loss = torch.sum(outputs['rgb'] * __grad_out_chunk)
                for v in reg_loss_dict.values():
                    __loss += torch.sum(v)
                self.model.aggregate_other_losses(reg_loss_dict)
            self.model.grad_scaler.scale(__loss).backward()  # type: ignore

        is_last = self.ddp_last()
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            grad_out_chunk = grad_out.view(-1, grad_out.shape[-1])[start_idx: end_idx]

            if (not is_last or end_idx < num_rays) and hasattr(self._model, 'no_sync'):
                with self._model.no_sync():
                    backward_iter(ray_bundle, grad_out_chunk)
            else:
                backward_iter(ray_bundle, grad_out_chunk)

        # unscale input if input required_grad. Because input is managed and optimized by Dr.Jit
        self.model.grad_scaler.unscale_(dummy_optimizer)
        if self.model.grad_scaler.is_enabled():
            optimizer_states = self.model.grad_scaler._per_optimizer_states  # noqa
            id_opt = id(dummy_optimizer)
            if id_opt in optimizer_states:
                found_inf = cast(
                    torch.Tensor,
                    sum([
                        t.to(self.device, non_blocking=True)
                        for t in optimizer_states[id_opt]["found_inf_per_device"].values()
                    ])
                )
                del optimizer_states[id_opt]
                if found_inf:
                    def clear_infinite(x):
                        invalid_mask = ~torch.isfinite(x)
                        x[invalid_mask] = 0
                        return int(invalid_mask.sum())

                    inf_cnt = clear_infinite(camera_ray_bundle.origins.grad)
                    inf_cnt += clear_infinite(camera_ray_bundle.directions.grad)
                    CONSOLE.log(f'[bold red]found {inf_cnt} inf in NeRF inputs, reset grad to 0...')

        loss = torch.sum(origins * camera_ray_bundle.origins.grad) + torch.sum(
            directions * camera_ray_bundle.directions.grad)
        if loss.requires_grad:
            loss.backward()

    def set_ddp_period(self):
        self._ddp_cd = self.opt_config.batch_size

    def ddp_last(self):
        if self._ddp_cd > 0:
            self._ddp_cd -= 1
        is_last = self._ddp_cd == 0
        return is_last

    def set_light_axis_angle(self, axis: List[float], angle: float) -> None:
        params = mi.traverse(self.sdf_scene.environment())
        params['to_world'] = mi.Transform4f.translate(0.5).rotate(axis=axis, angle=angle).translate(-0.5)
        params.update()

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[pathlib.Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        if not self.config.override_eval_images_metrics:
            return super().get_average_eval_image_metrics(step, output_path, get_std)
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.valid_indices)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            eval_cameras = self.datamanager.mi_train_dataset.cameras.to(self.device)
            for valid_index in self.datamanager.valid_indices:
                # time this the following line
                inner_start = time()
                batch = self.datamanager.cached_train[valid_index]
                height, width = batch['image'].shape[:2]
                num_rays = height * width
                outputs = self.render_camera_outputs(eval_cameras, batch['image_idx'])
                img_before = None
                if self.config.use_occlusion_image:
                    occlusion = mi.TensorXf(batch['occlusion'].float().to(self.device))
                    occlusion_mask = mi.TensorXf(batch['occlusion_mask'].float().to(self.device))
                    background = mi.TensorXf(batch['background'].float().to(self.device))
                    img = mi.TensorXf(outputs['rgb'])
                    mask = mi.TensorXf(outputs['mask'])
                    img = img + (1 - mask[..., None]) * background
                    img = occlusion + (1 - occlusion_mask) * img
                    img_before = outputs['rgb']
                    outputs['rgb'] = img.torch()
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if self.config.use_occlusion_image:
                    images_dict.update({
                        "before_occlusion_rgba": torch.cat([linear_to_srgb_torch(img_before), outputs['mask']], dim=-1)
                    })

                def imwrite(image_path, x):
                    if x.dtype in [np.float32, np.float64]:
                        x = (x * 255).astype(np.uint8)
                    iio.imwrite(image_path, x)

                if output_path is not None:
                    for key, val in images_dict.items():
                        imwrite(output_path / "{0:06d}-{1}.png".format(batch['image_idx'], key),
                                val.cpu().numpy())
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
