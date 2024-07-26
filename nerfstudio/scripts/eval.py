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

# !/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import mitsuba as mi
import yaml

mi.set_variant('cuda_ad_rgb')
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Override emitter scene path
    mesh_xml_path: Optional[Path] = None
    # Mesh XML Path, if enabled, will not use SDF but use mesh instead
    emitter_xml_path: Optional[Path] = None
    # Validate data to use when split=test
    test_data: Optional[Path] = None
    # The scale factor for scaling spatial data such as images, mask, semantics
    # along with relevant information about camera intrinsics
    camera_res_scale_factor: Optional[float] = None
    # Whether hide emitters when Mitsuba rendering
    hide_emitters: Optional[bool] = None
    # Multiply the image by the mask to mask out background.
    pre_mult_mask: Optional[bool] = None
    # Exclude mask loading
    load_mask: Optional[bool] = None
    # Only calculate loss at masked region
    eval_use_mask: Optional[bool] = None
    # Use test dataset's rotation id, Useful for evaluation
    use_test_rotations: Optional[bool] = None
    # Number of spp, to avoid MLE
    spp: Optional[int] = None
    # Data split for mi_train_dataset
    mi_data_split: Optional[Literal['train', 'eval', 'test']] = None
    # override the default get_average_eval_image_metrics
    override_eval_images_metrics: Optional[bool] = None
    # Reset all rotations as 0
    mock_zero_rotation: Optional[bool] = None
    # The method to use for splitting the dataset into train and eval.
    # Fraction splits based on a percentage for train and the remaining for eval.
    # Filename splits based on filenames containing train/eval.
    # Interval uses every nth frame for eval.
    # All uses all the images for any split.
    eval_mode: Optional[Literal["fraction", "filename", "interval", "all"]] = None
    # Perform load occlusion image before evaling
    load_occlusion: bool = False
    # override SAFE_EXP_MAX
    safe_exp_max: Optional[float] = None
    # mesh directory path passed to mitsuba xml to load mesh
    mesh_path: Optional[Path] = None

    def main(self) -> None:
        """Main function."""
        config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        if self.mesh_xml_path is not None:
            config.pipeline.mesh_xml_path = self.mesh_xml_path
        if self.emitter_xml_path is not None:
            config.pipeline.emitter_xml_path = self.emitter_xml_path
        if self.test_data is not None:
            config.pipeline.datamanager.dataparser.test_data = self.test_data
        if self.camera_res_scale_factor is not None:
            config.pipeline.datamanager.camera_res_scale_factor = self.camera_res_scale_factor
        if self.hide_emitters is not None:
            config.pipeline.hide_emitters = self.hide_emitters
        if self.pre_mult_mask is not None:
            config.pipeline.datamanager.pre_mult_mask = self.pre_mult_mask
        if self.load_mask is not None:
            config.pipeline.datamanager.dataparser.load_mask = self.load_mask
        if self.eval_use_mask is not None:
            config.pipeline.model.eval_use_mask = self.eval_use_mask
        if self.use_test_rotations is not None:
            config.pipeline.datamanager.use_test_rotations = self.use_test_rotations
        if self.spp is not None:
            config.pipeline.spp = self.spp
        if self.mi_data_split is not None:
            config.pipeline.datamanager.mi_data_split = self.mi_data_split
        if self.override_eval_images_metrics is not None:
            config.pipeline.override_eval_images_metrics = self.override_eval_images_metrics
        if self.mock_zero_rotation is not None:
            config.pipeline.datamanager.mock_zero_rotation = self.mock_zero_rotation
        if self.eval_mode is not None:
            config.pipeline.datamanager.dataparser.eval_mode = self.eval_mode
        if self.safe_exp_max is not None:
            from nerfstudio.fields import nerfacto_field
            nerfacto_field.SAFE_EXP_MAX = self.safe_exp_max
        if self.mesh_path is not None:
            config.pipeline.mesh_path = self.mesh_path
        config, pipeline, checkpoint_path, _ = eval_setup(config_path=None, _config=config)
        if self.load_occlusion:
            pipeline.load_occlusion(self.load_config.parent)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
