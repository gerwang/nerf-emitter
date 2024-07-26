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
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, List, Dict, Tuple

import torch
from torch.nn import Parameter
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.utils.colormaps import linear_to_srgb_torch


# Model related configs
@dataclass
class DummyModelConfig(ModelConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: DummyModel)
    hdr: bool = True
    """Use hdr in RGB Renderer"""
    eval_use_mask: bool = False
    """Only calculate loss at masked region"""


class DummyModel(Model):
    """Dummy model class
    Do nothing, as placeholder for mitsuba sdf optimization

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: DummyModelConfig

    def populate_modules(self):
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.mape = MeanAbsolutePercentageError()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        if self.config.hdr:
            srgb_predicted_rgb = linear_to_srgb_torch(predicted_rgb)
            srgb_gt_rgb = linear_to_srgb_torch(gt_rgb)
        else:
            srgb_predicted_rgb = predicted_rgb
            srgb_gt_rgb = gt_rgb
        srgb_gt_rgb = srgb_gt_rgb.clamp(0, 1)
        combined_rgb = torch.cat([srgb_gt_rgb, srgb_predicted_rgb], dim=1)

        masked_combined_rgb = None
        mask = None
        if self.config.eval_use_mask:
            assert 'mask' in batch
            mask = batch['mask'].to(self.device)
            srgb_gt_rgb = srgb_gt_rgb * mask
            srgb_predicted_rgb = srgb_predicted_rgb * mask
            predicted_rgb = predicted_rgb * mask
            gt_rgb = gt_rgb * mask
            masked_combined_rgb = torch.cat([srgb_gt_rgb, srgb_predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        def move_axis(x):
            return torch.moveaxis(x, -1, 0)[None, ...]

        gt_rgb = move_axis(gt_rgb)
        predicted_rgb = move_axis(predicted_rgb)
        srgb_gt_rgb = move_axis(srgb_gt_rgb)
        srgb_predicted_rgb = move_axis(srgb_predicted_rgb)

        if self.config.eval_use_mask:
            binary_mask = mask[..., 0] > 0.5
            psnr = self.psnr(srgb_gt_rgb[..., binary_mask], srgb_predicted_rgb[..., binary_mask])
        else:
            psnr = self.psnr(srgb_gt_rgb, srgb_predicted_rgb)
        ssim = self.ssim(srgb_gt_rgb, srgb_predicted_rgb)
        lpips = self.lpips(srgb_gt_rgb, srgb_predicted_rgb)

        mape = self.mape(predicted_rgb, gt_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "mape": float(mape)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, }
        if self.config.eval_use_mask:
            images_dict.update({"masked-img": masked_combined_rgb})
        return metrics_dict, images_dict
