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
crop_data.py
"""
from __future__ import annotations

import json
import os
from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
import shutil

import numpy as np

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.mi_gl_conversion import batch_affine_left
from nerfstudio.utils.poses import to4x4
from concurrent.futures import ProcessPoolExecutor


def get_projected_bbox(train_dataparser_outputs):
    cameras = train_dataparser_outputs.cameras
    aabb_vertices = torch.tensor([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ], dtype=torch.float32)
    aabb_vertices = batch_affine_left(to4x4(train_dataparser_outputs.dataparser_transform), aabb_vertices)
    aabb_vertices *= train_dataparser_outputs.dataparser_scale
    projected_aabb_coords, _ = project(
        to4x4(cameras.camera_to_worlds),
        cameras.get_intrinsics_matrices(),
        aabb_vertices,
    )
    # min_x, max_x, min_y, max_y = get_crop_size_torch(projected_aabb_coords)
    x_coords = projected_aabb_coords[..., 0]
    y_coords = projected_aabb_coords[..., 1]
    min_x = x_coords.min(dim=-1).values
    max_x = x_coords.max(dim=-1).values
    min_y = y_coords.min(dim=-1).values
    max_y = y_coords.max(dim=-1).values
    valid_mask = (min_x >= 0) & (max_x <= cameras.width[..., 0]) & (min_y >= 0) & (max_y <= cameras.height[..., 0])
    half_side_length = torch.maximum(max_y - min_y, max_x - min_x) * 0.5
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    min_x = (center_x - half_side_length).floor().long()
    max_x = min_x + (2 * half_side_length).ceil().long()
    min_y = (center_y - half_side_length).floor().long()
    max_y = min_y + (2 * half_side_length).ceil().long()
    valid_mask &= (min_x >= 0) & (max_x <= cameras.width[..., 0]) & (min_y >= 0) & (max_y <= cameras.height[..., 0])
    side_length_for_invalid = torch.minimum(cameras.width[~valid_mask, 0], cameras.height[~valid_mask, 0])
    min_x[~valid_mask] = 0
    max_x[~valid_mask] = side_length_for_invalid
    min_y[~valid_mask] = 0
    max_y[~valid_mask] = side_length_for_invalid
    return max_x, max_y, min_x, min_y, valid_mask


def undistort_crop_resize_image(K, distortion_params, image_filename, image_output_path, this_max_x,
                                this_max_y, this_min_x, this_min_y, res):
    distortion_params = np.array(
        [
            distortion_params[0],
            distortion_params[1],
            distortion_params[4],
            distortion_params[5],
            distortion_params[2],
            distortion_params[3],
            0,
            0,
        ]
    )
    image = cv2.imread(str(image_filename), cv2.IMREAD_UNCHANGED)
    image = cv2.undistort(image, K, distortion_params)
    image = image[this_min_y:this_max_y, this_min_x:this_max_x]
    image = cv2.resize(image, (res, res), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(image_output_path / image_filename.name), image)


@dataclass
class CropData:
    """Generate data of an outer scene and an inner object."""

    # output directory
    output_path: Path
    # Specifies the dataparser used to unpack the data.
    dataparser_config: AnnotatedDataParserUnion = field(default_factory=lambda: NerfstudioDataParserConfig())
    # resolution width and height, only supports square
    res: int = 512

    def processed_json(self, train_dataparser_outputs: DataparserOutputs, valid_mask: torch.Tensor):
        res = {
            'camera_model': 'PINHOLE',
        }
        frames = []
        for index in range(len(train_dataparser_outputs.cameras)):
            frame = {
                'file_path': str(
                    train_dataparser_outputs.image_filenames[index].relative_to(self.dataparser_config.data)),
                'w': int(train_dataparser_outputs.cameras.width[index]),
                'h': int(train_dataparser_outputs.cameras.height[index]),
                'fl_x': float(train_dataparser_outputs.cameras.fx[index]),
                'fl_y': float(train_dataparser_outputs.cameras.fy[index]),
                'cx': float(train_dataparser_outputs.cameras.cx[index]),
                'cy': float(train_dataparser_outputs.cameras.cy[index]),
                'transform_matrix': to4x4(train_dataparser_outputs.cameras.camera_to_worlds[index]).tolist(),
                'valid': bool(valid_mask[index]),
            }
            if 'rotations' in train_dataparser_outputs.metadata:
                frame['rotation'] = str(int(train_dataparser_outputs.metadata['rotations'][index]))
            frames.append(frame)
        res['frames'] = frames
        if 'rotation_transform_matrices' in train_dataparser_outputs.metadata:
            res['rotations'] = {
                str(int(k)): v.tolist() for k, v in
                train_dataparser_outputs.metadata['rotation_transform_matrices'].items()
            }
        if 'rotation_aabb' in train_dataparser_outputs.metadata:
            res['rotation_aabb'] = train_dataparser_outputs.metadata['rotation_aabb'].tolist()
        return res

    def main(self) -> None:
        """Main function."""
        dataparser = self.dataparser_config.setup()
        train_dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")

        max_x, max_y, min_x, min_y, valid_mask = get_projected_bbox(train_dataparser_outputs)

        cameras = train_dataparser_outputs.cameras

        image_output_path = self.output_path / 'images'
        if image_output_path.exists():
            shutil.rmtree(image_output_path)
        image_output_path.mkdir(parents=True, exist_ok=True)

        all_distortion_params = cameras.distortion_params.cpu().numpy()
        all_K = cameras.get_intrinsics_matrices().cpu().numpy()
        cameras.distortion_params = None
        cameras.crop_output_resolution(min_x, min_y, max_x - min_x, max_y - min_y)
        cameras.rescale_output_resolution(self.res / (max_x - min_x).float())

        with open(self.output_path / 'transforms.json', "w", encoding="utf-8") as f:
            json.dump(self.processed_json(train_dataparser_outputs, valid_mask), f, indent=4)

        with ProcessPoolExecutor() as executor:
            futures = []
            for index in range(len(cameras)):
                distortion_params = all_distortion_params[index]
                K = all_K[index]
                image_filename = train_dataparser_outputs.image_filenames[index]
                this_min_y = int(min_y[index])
                this_max_y = int(max_y[index])
                this_min_x = int(min_x[index])
                this_max_x = int(max_x[index])
                futures.append(executor.submit(undistort_crop_resize_image, K, distortion_params, image_filename,
                                               image_output_path, this_max_x, this_max_y, this_min_x, this_min_y,
                                               self.res))
            for future in tqdm(futures):
                future.result()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(CropData).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(CropData)  # noqa


def project(
        c2w: Float[Tensor, "batch 4 4"],
        K: Float[Tensor, "batch 3 3"],
        points: Float[Tensor, "N 3"],
) -> Tuple[Float[Tensor, "batch N 2"], Float[Tensor, "batch N 1"]]:
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), device=points.device)], dim=-1)  # [N, 4]
    w2c = torch.inverse(c2w)  # [batch, 4, 4]
    cam_points = w2c[..., None, :, :] @ points[None, ..., None]  # [batch, N, 4, 1]
    cam_points = cam_points[..., :3, :]  # [batch, N, 3, 1]
    depth = torch.linalg.norm(cam_points, dim=-2)  # [batch, N, 1]
    # Actually, these are **distance** rather than **depth**. However, both nerfstudio and mitsuba confuse
    cam_points[..., 1:, :] *= -1
    image_coords = K.unsqueeze(1) @ (cam_points / cam_points[..., 2:, :])  # [batch, N, 3, 1]
    image_coords = image_coords[..., :2, 0]
    return image_coords, depth
