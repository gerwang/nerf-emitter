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
Nerfacto with Mitsuba SDF representing the inner region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type, List, Literal

import torch
from torch.autograd import forward_ad
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared, \
    interlevel_loss, distortion_loss, orientation_loss, pred_normal_loss
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class SdfNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: SdfNerfactoModel)
    depth_strategy: Literal['contrib', 'expected', 'median'] = 'contrib'


class SdfNerfactoModel(NerfactoModel):
    """Nerfacto with Mitsuba SDF representing the inner region.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SdfNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.renderer_pc_depth = DepthRenderer(method=self.config.depth_strategy)

    def get_generators(self) -> List[torch.Generator]:
        return [
            self.proposal_sampler.initial_sampler.get_generator(self.device),
        ] + [
            x.get_generator(self.device) for x in self.proposal_sampler.pdf_samplers
        ]

    def get_rgb(self, ray_bundle: RayBundle) -> torch.Tensor:

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        if self.rotater is not None:
            ray_samples.camera_indices = self.rotater.map_rotation_ids(ray_samples.camera_indices)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        return rgb

    def forward_grad(self, ray_bundle: RayBundle, grad_o: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        if self.rotater is not None:
            ray_samples.camera_indices = self.rotater.map_rotation_ids(ray_samples.camera_indices)

        with forward_ad.dual_level():
            ray_samples.frustums.origins = forward_ad.make_dual(
                ray_samples.frustums.origins,
                grad_o.unsqueeze(-2).expand_as(ray_samples.frustums.origins)
            )
            ray_samples.frustums.directions = forward_ad.make_dual(
                ray_samples.frustums.directions,
                grad_v.unsqueeze(-2).expand_as(ray_samples.frustums.directions)
            )
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            grad_rgb = forward_ad.unpack_dual(rgb).tangent

        return grad_rgb

    def get_backward_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        # only attach ray_bundle in the ray_samples, detach them in ray_samples_list
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        ray_bundle.origins = ray_bundle.origins.detach()
        ray_bundle.directions = ray_bundle.directions.detach()
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        if self.rotater is not None:
            ray_samples.camera_indices = self.rotater.map_rotation_ids(ray_samples.camera_indices)
        ray_samples = ray_samples[...]
        ray_samples.frustums.origins = origins[..., None, :].expand_as(ray_samples.frustums.origins)
        ray_samples.frustums.directions = directions[..., None, :].expand_as(ray_samples.frustums.directions)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        outputs = {
            "rgb": rgb,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        return outputs

    def get_reg_loss_dict(self, outputs):
        loss_dict = {}
        loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
            outputs["weights_list"], outputs["ray_samples_list"]
        )
        loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(outputs["weights_list"],
                                                                                          outputs["ray_samples_list"])
        if self.config.predict_normals:
            # orientation loss for computed normals
            loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                outputs["rendered_orientation_loss"]
            )

            # ground truth supervision for normals
            loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                outputs["rendered_pred_normal_loss"]
            )
        return loss_dict

    def get_point_lights(self, ray_bundle: RayBundle):

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        if self.config.far_intersect:
            ray_bundle = self.far_intersect_collider(ray_bundle)

        ray_samples: RaySamples
        ray_samples, _, _ = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        if self.rotater is not None:
            ray_samples.camera_indices = self.rotater.map_rotation_ids(ray_samples.camera_indices)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=self.device, dtype=torch.float32)
        brightness_grad, lum, rgb_detach, weights = self.calc_brightness_grad(ray_bundle, ray_samples, rgb_weights)
        depth = self.renderer_pc_depth(weights=weights, ray_samples=ray_samples, values=lum)

        # if self.field.spatial_distortion is not None: # TODO implement kmeans in contracted space
        #     pos_samples = self.field.spatial_distortion(pos_samples)

        outputs = {
            "rgb": rgb_detach,
            "depth": depth,
            "brightness_grad": brightness_grad,
        }
        return outputs
