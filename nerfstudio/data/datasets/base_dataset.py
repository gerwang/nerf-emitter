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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from skimage.transform import resize
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.utils.colormaps import srgb_to_linear, linear_to_srgb


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras
    use_cache: bool = True
    cache_dict: dict = {}

    @classmethod
    def cached_imread(cls, image_filename):
        import mitsuba as mi
        if mi.variant() is None:
            mi.set_variant('scalar_rgb')
        image = None
        if cls.use_cache:
            if image_filename in cls.cache_dict:
                image = cls.cache_dict[image_filename]
        if image is None:
            image = np.asarray(mi.Bitmap(str(image_filename)))
            image[~np.isfinite(image)] = 0
        if cls.use_cache:
            if image_filename not in cls.cache_dict:
                cls.cache_dict[image_filename] = image
        return image

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, pre_mult_mask: bool = False):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.is_hdr = dataparser_outputs.is_hdr
        self.to_linear = dataparser_outputs.to_linear
        self.tone_mapping = dataparser_outputs.tone_mapping
        self.pre_mult_mask = pre_mult_mask

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.float32]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        image = self.cached_imread(image_filename)
        if self.pre_mult_mask and self._dataparser_outputs.mask_filenames is not None:
            mask_filename = self._dataparser_outputs.mask_filenames[image_idx]
            mask = self.cached_imread(mask_filename)
            mask = mask.astype(np.float32) / np.iinfo(mask.dtype).max
            image *= mask
        if self.scale_factor != 1.0:
            height, width = image.shape[:2]
            newsize = (int(height * self.scale_factor), int(width * self.scale_factor))
            image = resize(image, newsize, order=1, mode='reflect', anti_aliasing=True)
        old_dtype = image.dtype
        image = np.array(image, dtype=np.float32)  # shape is (h, w) or (h, w, 3 or 4)
        if old_dtype == np.uint8:
            image = image / 255.0
        elif old_dtype == np.uint16:
            image = image / 65535.0
        else:
            assert old_dtype in [np.float16, np.float32, np.float64]
        if self.to_linear:
            assert not self.is_hdr, "To linear can only be used when LDR"
            image = srgb_to_linear(image)
        if self.tone_mapping:
            assert self.is_hdr, "Tone mapping can only be used when HDR"
            image = linear_to_srgb(image)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.float32
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            mask = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                    mask.shape[:2] == image.shape[:2]
            ), f"Mask and image have different shapes. Got {mask.shape[:2]} and {image.shape[:2]}"
            if self.pre_mult_mask:
                image *= mask
            else:
                data["mask"] = mask
        data['image'] = image
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
