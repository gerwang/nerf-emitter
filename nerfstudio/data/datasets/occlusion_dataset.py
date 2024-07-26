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
Occlusion dataset.
"""

from typing import Dict

import torch
from skimage.transform import resize

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class OcclusionDataset(InputDataset):
    """Dataset that returns foreground and background images.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, pre_mult_mask: bool = False):
        super().__init__(dataparser_outputs, scale_factor, pre_mult_mask)
        self.occlusion_filenames = dataparser_outputs.metadata.get('occlusion_filenames', None)
        self.background_filenames = dataparser_outputs.metadata.get('background_filenames', None)

    def get_metadata(self, data: Dict) -> Dict:
        res = {}
        if self.occlusion_filenames is not None:
            filepath = self.occlusion_filenames[data["image_idx"]]
            occlusion, occlusion_mask = self.load_occlusion(filepath)
            res.update({
                'occlusion': occlusion,
                'occlusion_mask': occlusion_mask,
            })
        if self.background_filenames is not None:
            filepath = self.background_filenames[data["image_idx"]]
            background, background_mask = self.load_occlusion(filepath)
            res.update({
                'background': background,
                # 'background_mask': background_mask,
            })
        return res

    def load_occlusion(self, filepath):
        image = self.cached_imread(filepath)
        if self.scale_factor != 1.0:
            height, width = image.shape[:2]
            newsize = (int(height * self.scale_factor), int(width * self.scale_factor))
            image = resize(image, newsize, order=1, mode='reflect', anti_aliasing=True)
        image = torch.from_numpy(image).float()
        occlusion, occlusion_mask = image.split([3, 1], dim=-1)
        return occlusion, occlusion_mask
