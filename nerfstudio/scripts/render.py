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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import os
from abc import abstractmethod

import mitsuba as mi
import trimesh
import yaml

from nerfstudio.configs.base_config import MachineConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import CropMode
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.train import launch, _set_random_seed
from nerfstudio.utils.mi_gl_conversion import get_nerfstudio_matrix, torch_point_scale_shifted_gl2mi
from nerfstudio.utils.render_utils import CropData, render_trajectory_video

mi.set_variant('cuda_ad_rgb')
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import tyro
from typing_extensions import Annotated
from emitters.util import affine_left

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    get_spiral_path,
    get_blender_test_path,
)
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks, comms
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.scripts import run_command

from rich import box, style
from rich.panel import Panel
from rich.table import Table

def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )


def main(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    assert hasattr(config, 'base_render')
    base_render: BaseRender = getattr(config, 'base_render')
    _, pipeline, _, _ = eval_setup(
        config_path=None, _config=config,
        local_rank=local_rank, world_size=world_size,
        test_mode=base_render.test_mode
    )
    base_render.main(pipeline)


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png", "exr"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    test_mode: Literal['inference', 'test'] = 'test'
    """test_mode for eval_setup"""
    machine: MachineConfig = field(default_factory=MachineConfig)
    """Machine configuration"""
    mesh_xml_path: Optional[Path] = None
    """Mesh XML Path, if enabled, will not use SDF but use mesh instead"""
    camera_xml_path: Optional[Path] = None
    """Override camera path with mitsuba xml"""
    emitter_xml_path: Optional[Path] = None
    """Override emitter scene path"""
    spp: Optional[int] = None
    """Number of spp, to avoid MLE"""
    spp_per_batch: Optional[int] = None
    """Number of spp per batch, to avoid MLE"""
    primal_spp_mult: Optional[int] = None
    """Override primal spp multiplier"""
    guiding_type: Optional[Literal['vmf', 'env', 'emitter_xml']] = None
    """Path guiding type"""
    mi_config_name: Optional[str] = None
    """Method to be used for the optimization. Default: warpone"""
    sdf_position: Optional[List[float]] = None
    """Specify sdf position"""
    sdf_cube_size: Optional[float] = None
    """Override sdf voxel with a cube"""
    padding_size: Optional[float] = None
    """Bounding box padding size"""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    num_nerf_samples_per_ray: Optional[int] = None
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Optional[int] = None
    """Number of samples per ray for each proposal network."""
    hide_emitters: Optional[bool] = None
    """Whether hide emitters when Mitsuba rendering"""
    power_of_two: Optional[bool] = None
    """Whether it performs M = ⌊log2(N + 1)⌋ iterations with power-of-two sample counts. 
    This approach was initially proposed by Müller et al. [2017] to limit the impact 
    of initial high-variance estimates on the final image."""
    use_visibility: Optional[bool] = None
    """Whether it uses visibility in direct integrator"""
    disable_rotater: Optional[bool] = False
    """Disable rotater when evaluation is performed"""
    use_nerf_render: Optional[bool] = False
    """Use NeRF's render_camera_outputs instead of Mitsuba"""
    visualize_xyz_gradient: Optional[bool] = False
    """Visualize xyz gradient, should be disabled in training"""
    mock_zero_rotation: Optional[bool] = None
    """Reset all rotations as 0"""
    eval_mode: Optional[Literal["fraction", "filename", "interval", "all"]] = None
    """The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split."""
    mesh_path: Optional[Path] = None
    """mesh directory path passed to mitsuba xml to load mesh"""
    mesh_type: Optional[Literal['diffuse', 'principled']] = None
    """mesh type in mesh_path"""
    safe_exp_max: Optional[float] = None
    """override SAFE_EXP_MAX"""
    test_data: Optional[Path] = None
    """Validate data to use when split=test"""
    eval_mode: Optional[Literal["fraction", "filename", "interval", "all"]] = None
    """The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split."""
    camera_optimizer_mode: Optional[Literal["off", "SO3xR3", "SE3"]] = None
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""
    clear_datamanager_load_dir: bool = False
    """Clear datamanager load_dir"""
    denoise: Optional[bool] = None
    """Use Optix denoiser when render_camera_outputs"""
    camera_res_scale_factor: Optional[float] = None
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics"""
    mock_split: Optional[str] = None
    """Mock split to use another split for testing"""

    @abstractmethod
    def main(self, pipeline: Pipeline) -> None:
        """
        Do actual rendering here
        """

    def entry_point(self) -> None:
        config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)

        config.machine = self.machine
        config.base_render = self
        if self.mesh_xml_path is not None:
            config.pipeline.mesh_xml_path = self.mesh_xml_path
        if self.emitter_xml_path is not None:
            config.pipeline.emitter_xml_path = self.emitter_xml_path
        if self.spp is not None:
            config.pipeline.spp = self.spp
        if self.spp_per_batch is not None:
            config.pipeline.spp_per_batch = self.spp_per_batch
        if self.primal_spp_mult is not None:
            config.pipeline.primal_spp_mult = self.primal_spp_mult
        if self.guiding_type is not None:
            config.pipeline.guiding_type = self.guiding_type
        if self.mi_config_name is not None:
            config.pipeline.mi_config_name = self.mi_config_name
        if self.sdf_position is not None:
            config.pipeline.sdf_position = self.sdf_position
        if self.sdf_cube_size is not None:
            config.pipeline.sdf_cube_size = self.sdf_cube_size
        if self.padding_size is not None:
            config.pipeline.padding_size = self.padding_size
        if self.data is not None:
            config.pipeline.datamanager.data = self.data
        if self.num_nerf_samples_per_ray is not None:
            config.pipeline.model.num_nerf_samples_per_ray = self.num_nerf_samples_per_ray
        if self.num_proposal_samples_per_ray is not None:
            config.pipeline.model.num_proposal_samples_per_ray = (
                self.num_proposal_samples_per_ray * 2, self.num_proposal_samples_per_ray * 2)
        if self.hide_emitters is not None:
            config.pipeline.hide_emitters = self.hide_emitters
        if self.power_of_two is not None:
            config.pipeline.power_of_two = self.power_of_two
        if self.use_visibility is not None:
            config.pipeline.use_visibility = self.use_visibility
        if self.disable_rotater is not None:
            config.pipeline.datamanager.disable_rotater = self.disable_rotater
        if self.use_nerf_render is not None:
            config.pipeline.use_nerf_render = self.use_nerf_render
        if self.visualize_xyz_gradient is not None:
            config.pipeline.model.visualize_xyz_gradient = self.visualize_xyz_gradient
        if self.eval_mode is not None:
            config.pipeline.datamanager.dataparser.eval_mode = self.eval_mode
        if self.mock_zero_rotation is not None:
            config.pipeline.datamanager.mock_zero_rotation = self.mock_zero_rotation
        if self.mesh_path is not None:
            config.pipeline.mesh_path = self.mesh_path
        if self.mesh_type is not None:
            config.pipeline.mesh_type = self.mesh_type
        if self.safe_exp_max is not None:
            from nerfstudio.fields import nerfacto_field
            nerfacto_field.SAFE_EXP_MAX = self.safe_exp_max
        if self.test_data is not None:
            config.pipeline.datamanager.dataparser.test_data = self.test_data
        if self.eval_mode is not None:
            config.pipeline.datamanager.dataparser.eval_mode = self.eval_mode
        if self.camera_optimizer_mode is not None:
            config.pipeline.datamanager.camera_optimizer.mode = self.camera_optimizer_mode
        if self.clear_datamanager_load_dir:
            config.pipeline.datamanager.load_dir = None
        if self.denoise is not None:
            config.pipeline.denoise = self.denoise
        if self.camera_res_scale_factor is not None:
            config.pipeline.datamanager.camera_res_scale_factor = self.camera_res_scale_factor
        if self.mock_split is not None:
            config.pipeline.datamanager.dataparser.mock_split = self.mock_split

        launch(
            main_func=main,
            num_devices_per_machine=config.machine.num_devices,
            device_type=config.machine.device_type,
            num_machines=config.machine.num_machines,
            machine_rank=config.machine.machine_rank,
            dist_url=config.machine.dist_url,
            config=config,
        )


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    test_mode: Literal['inference', 'test'] = 'inference'
    """test_mode for eval_setup"""

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras

        seconds = self.interpolation_steps * len(cameras) / self.frame_rate
        camera_path = get_interpolated_camera_path(
            cameras=cameras,
            steps=self.interpolation_steps,
            order_poses=self.order_poses,
        )

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


@dataclass
class SpiralRender(BaseRender):
    """Render a spiral trajectory (often not great)."""

    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.1
    """Radius of the spiral."""

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()

        assert isinstance(pipeline.datamanager, VanillaDataManager)
        steps = int(self.frame_rate * self.seconds)
        camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
        camera_path = get_spiral_path(camera_start, steps=steps, radius=self.radius)

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


@dataclass
class BlenderRender(BaseRender):
    """Render a blender trajectory that revolves around the central object."""

    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()

        assert isinstance(pipeline.datamanager, VanillaDataManager)
        steps = int(self.frame_rate * self.seconds)
        camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
        camera_path = get_blender_test_path(camera_start, num_views=steps)

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


@dataclass
class EvalRender(BaseRender):
    """Render a trajectory that replicates the eval dataset."""

    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    build_proposal: bool = False
    """Whether it invokes build_emitter_proposal"""
    save_transforms: bool = False
    """Save camera transforms"""
    hide_sdf: bool = False
    """Hide sdf in rendering, currently used to render environment map"""
    use_mi_train: bool = False
    """Use mi_train_dataset instead of eval_dataset"""
    mock_aabb_scale: Optional[float] = None
    """Mock the model's aabb, to fix cornell"""
    disable_custom_dataparser: bool = True
    """disable custom dataparser_config"""
    dataparser_config: AnnotatedDataParserUnion = field(default_factory=lambda: NerfstudioDataParserConfig())
    """Specifies the dataparser used to load the camera"""
    render_occlusion: bool = False
    occlusion_dir: Optional[Path] = None

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        if self.build_proposal:
            PRETRAIN_ITER = 2000
            MI_OPT_ITER = 320
            pipeline.rescale_train_dataset(PRETRAIN_ITER)
            pipeline.build_emitter_proposal()
            pipeline.rescale_train_dataset(PRETRAIN_ITER + MI_OPT_ITER - 1)
        if self.hide_sdf:
            if comms.is_main_process():
                pipeline.sdf_object = pipeline.sdf_scene.integrator().sdf = None
        if self.mock_aabb_scale is not None and hasattr(pipeline.model, 'mock_aabb'):
            pipeline.model.mock_aabb(self.mock_aabb_scale)

        install_checks.check_ffmpeg_installed()

        assert isinstance(pipeline.datamanager, VanillaDataManager)

        target_dataset = pipeline.datamanager.eval_dataset
        if self.use_mi_train:
            target_dataset = pipeline.datamanager.mi_train_dataset
        if self.camera_xml_path is not None:
            camera_path = mi.load_file(str(self.camera_xml_path)).sensors()
            image_names = None
            if self.save_transforms:
                res = {
                    'frames': [],
                }
                sensor = None
                for i, sensor in enumerate(camera_path):
                    frame = {
                        'file_path': f'{i:04d}.exr',
                        'transform_matrix': get_nerfstudio_matrix(sensor, scale=True),
                    }
                    res['frames'].append(frame)
                res.update({
                    'x_fov': mi.traverse(sensor)['x_fov'][0],
                    'w': sensor.film().size().x,
                    'h': sensor.film().size().y,
                    'aabb_scale': pipeline.datamanager.train_dataparser_outputs.dataparser_scale * 2,
                })
                if not self.output_path.exists():
                    self.output_path.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(os.path.join(self.output_path, 'transforms.json'), 'w'), indent=2)
        else:
            if not self.disable_custom_dataparser:
                dataparser = self.dataparser_config.setup()
                train_dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")
                camera_path = train_dataparser_outputs.cameras
                image_names = [x.stem for x in train_dataparser_outputs.image_filenames]
            else:
                camera_path = target_dataset.cameras
                image_names = [x.stem for x in
                               getattr(target_dataset, '_dataparser_outputs').image_filenames]

        if self.render_occlusion:
            assert self.occlusion_dir is not None
            pipeline.render_occlusion(self.occlusion_dir / 'occlusion_images', CropMode.NEAR2INF, cameras=camera_path)
            pipeline.render_occlusion(self.occlusion_dir / 'background_images', CropMode.FAR, cameras=camera_path)
        occlusion_image_paths = None
        background_image_paths = None
        if self.occlusion_dir is not None:
            occlusion_image_paths = [self.occlusion_dir / 'occlusion_images' / f'{i:05d}.exr' for i in
                                     range(len(camera_path))]
            background_image_paths = [self.occlusion_dir / 'background_images' / f'{i:05d}.exr' for i in
                                      range(len(camera_path))]

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=len(camera_path) / self.frame_rate,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            image_names=image_names,
            occlusion_image_paths=occlusion_image_paths,
            background_image_paths=background_image_paths,
        )


@dataclass
class RotateLightRender(BaseRender):
    """Render a trajectory that replicates the eval dataset."""

    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    eval_idx: int = 0
    """The camera pose is selected from one evaluation view"""
    rotate_axis: List[float] = field(default_factory=lambda: [0., 1., 0.])

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()

        assert isinstance(pipeline.datamanager, VanillaDataManager)
        n_frame = int(self.seconds * self.frame_rate)
        camera_path = pipeline.datamanager.eval_dataset.cameras[
            torch.full((n_frame,), fill_value=self.eval_idx,
                       device=pipeline.datamanager.eval_dataset.cameras.device)]
        angles = np.linspace(0, 360, n_frame + 1)[:-1]
        axis_angles = [(self.rotate_axis, angle) for angle in angles]

        render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=len(camera_path) / self.frame_rate,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            axis_angles=axis_angles,
        )


@dataclass
class StrokeToCameraXml(BaseRender):
    """Generate data of an outer scene and an inner object."""

    # input coordinate txt file
    coord_path: Path = Path('path_coordinates.txt')
    # Specifies the dataparser used to load the camera
    dataparser_config: AnnotatedDataParserUnion = field(default_factory=lambda: NerfstudioDataParserConfig())
    # camera index
    camera_idx: int = 0
    # camera index of the corresponding filename
    camera_name: Optional[str] = None
    # envmap width
    envmap_resx: int = 1024
    # envmap height
    envmap_resy: int = 512
    # disable custom dataparser_config
    disable_custom_dataparser: bool = False

    def main(self, pipeline: Pipeline) -> None:
        """Main function."""
        if self.disable_custom_dataparser:
            train_dataparser_outputs = pipeline.datamanager.mi_train_dataparser_outputs
        else:
            dataparser = self.dataparser_config.setup()
            train_dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")

        coordinates = torch.from_numpy(np.loadtxt(self.coord_path, dtype=np.int32)).long().to(pipeline.device)

        scale_mi2gl = 2 * pipeline.scene_scale

        cameras = train_dataparser_outputs.cameras
        if self.camera_name is not None:
            image_filenames = train_dataparser_outputs.image_filenames
            self.camera_idx = [i for i in range(len(image_filenames)) if image_filenames[i].stem == self.camera_name][0]
        sensor = pipeline.get_mi_sensor(cameras, self.camera_idx)

        # render a depth map from the camera
        img = mi.render(
            pipeline.sdf_scene,
            spp=8,
            sensor=sensor,
            integrator=pipeline.depth_integrator,
        )
        depth_image = img[..., -1] + sensor.near_clip()  # [height, n_img*width]
        depth_image = depth_image.torch().unsqueeze(-1).float() * scale_mi2gl
        # convert the depth map a point cloud map, in Mitsuba space
        camera_ray_bundle = cameras.generate_rays(self.camera_idx).to(pipeline.device)
        point_image = camera_ray_bundle.origins + depth_image * camera_ray_bundle.directions
        point_image = affine_left(pipeline.torch_gl2mi_left, point_image)
        point_image = torch_point_scale_shifted_gl2mi(point_image, pipeline.scene_scale)
        # create spherical camera for each point
        path_points = point_image[coordinates[:, 1], coordinates[:, 0]]
        path_points = path_points.cpu().numpy()
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        trimesh.PointCloud(path_points).export(self.output_path / 'path_points.ply')
        scene_dict = {
            'type': 'scene'
        }
        for i in range(path_points.shape[0]):
            scene_dict[f'sensor_{i}'] = {
                'type': 'spherical',
                'film': {
                    'type': 'hdrfilm',
                    'width': self.envmap_resx,
                    'height': self.envmap_resy,
                    'filter': {
                        'type': 'gaussian',
                        'stddev': 0.25,
                    },
                    'pixel_format': 'rgba',
                },
                'sampler': {
                    'type': 'independent',
                },
                'near_clip': 0,
                'to_world': mi.ScalarTransform4f.translate(path_points[i]),
            }
        # save spherical cameras to xml
        mi.xml.dict_to_xml(scene_dict, str(self.output_path / 'cameras.xml'))
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Path Points", str(self.output_path / 'path_points.ply'))
        table.add_row("Camera XML", str(self.output_path / 'cameras.xml'))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Export XML Complete :tada:[/bold]", expand=False))


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[SpiralRender, tyro.conf.subcommand(name="spiral")],
        Annotated[BlenderRender, tyro.conf.subcommand(name="blender")],
        Annotated[EvalRender, tyro.conf.subcommand(name="eval")],
        Annotated[RotateLightRender, tyro.conf.subcommand(name="rotate-light")],
        Annotated[StrokeToCameraXml, tyro.conf.subcommand(name="stroke")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).entry_point()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
