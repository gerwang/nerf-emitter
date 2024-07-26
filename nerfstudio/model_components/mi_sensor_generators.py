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
Mitsuba sensor generator.
"""
from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from torch import nn

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.mi_gl_conversion import gl2mi_right, gl2mi_left, torch_scale_shifted_gl2mi


class MitsubaSensorGenerator(nn.Module):

    def __init__(self, cameras: Cameras, scene_scale: float, pose_optimizer: CameraOptimizer | None = None,
                 patch_width: int | None = None, patch_height: int | None = None,
                 filter_type: str = 'gaussian') -> None:
        super().__init__()
        self.cameras = cameras
        self.pose_optimizer = pose_optimizer
        self.patch_width = patch_width if patch_width is not None else int(cameras.width[0])
        self.patch_height = patch_height if patch_height is not None else int(cameras.height[0])
        self.film = mi.load_dict({
            'type': 'hdrfilm', 'width': self.patch_width, 'height': self.patch_height,
            'pixel_format': 'rgba', 'pixel_filter': {'type': filter_type}, 'sample_border': True})
        self.sampler = mi.load_dict({'type': 'independent'})
        self.gl2mi_left_torch = torch.from_numpy(gl2mi_left).float().to(cameras.device)
        self.gl2mi_right_torch = torch.from_numpy(gl2mi_right).float().to(cameras.device)
        self.scene_scale = scene_scale

    def get_x_fov(self, camera):
        x_fov = np.rad2deg(2 * np.arctan2(0.5 * self.patch_width, float(camera.fx)))
        return x_fov

    def get_mi_sensor(self, camera, crop_x: int, crop_y: int):
        """Generates regularly spaced sensors for optimization. Returns a list of Mitsuba sensors"""
        s = mi.load_dict({
            'type': 'perspective',
            'fov': self.get_x_fov(camera),
            'sampler': self.sampler,
            'film': self.film,
            'principal_point_offset_x': (crop_x + 0.5 * self.patch_width - float(camera.cx)) / self.patch_width,
            'principal_point_offset_y': (crop_y + 0.5 * self.patch_height - float(camera.cy)) / self.patch_height,
        })
        params = mi.traverse(s)
        affine_c2w = torch.cat(
            [camera.camera_to_worlds,
             torch.tensor([[0., 0., 0., 1.]], dtype=camera.camera_to_worlds.dtype, device=camera.device)])
        affine_c2w = self.gl2mi_left_torch @ affine_c2w @ self.gl2mi_right_torch
        affine_c2w = torch_scale_shifted_gl2mi(affine_c2w, self.scene_scale)
        params['to_world'] = mi.Transform4f(dr.unravel(mi.Matrix4f, mi.TensorXf(affine_c2w.t().float().to('cuda:0'))))
        params.update()
        return s

    def forward(self, c: int) -> mi.Sensor:
        """Index into the cameras to generate the rays.

        Args:
            c: camera index
        """
        c_tensor = torch.tensor([c])
        with torch.no_grad():
            # PyTorch camera cannot optimize here, it is already optimizer during the NeRF pretraining phase
            camera_opt_to_camera = self.pose_optimizer(c_tensor)

        camera = self.cameras[c_tensor]
        camera.camera_to_worlds = pose_utils.multiply(camera.camera_to_worlds, camera_opt_to_camera)  # (1, 3, 4)

        sensor = self.get_mi_sensor(camera[0], 0, 0)
        return sensor
