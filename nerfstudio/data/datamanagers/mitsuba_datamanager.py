# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Data manager that can return mitsuba sensor and GTs
"""

from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Type, Tuple, Dict, Literal, Optional, Generic, get_origin, get_args, ForwardRef, cast

import mitsuba as mi
import torch
from rich.progress import track

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.model_components.mi_sensor_generators import MitsubaSensorGenerator
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class MitsubaDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: MitsubaDataManager)
    cache_images: Literal["no-cache", "cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "numpy", caches as numpy arrays, if "torch", caches as torch tensors."""
    mi_data_split: Literal['train', 'eval', 'test'] = 'train'
    """Data split for mi_train_dataset"""


class MitsubaDataManager(VanillaDataManager, Generic[TDataset]):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: MitsubaDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_record_step = -1
        self.rays_per_step = 0

        self.mi_train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(
            split=f"mi_{self.config.mi_data_split}")
        self.mi_train_dataset = self.create_mi_train_dataset()
        self.setup_mitsuba_train()

        if 'valid_mask' in self.mi_train_dataparser_outputs.metadata:
            valid_mask = self.mi_train_dataparser_outputs.metadata['valid_mask']
            self.valid_indices = [i for i in range(len(self.mi_train_dataset)) if valid_mask[i]]
        else:
            self.valid_indices = [i for i in range(len(self.mi_train_dataset))]
        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [x for x in self.valid_indices]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

    def cache_images(self, cache_images_option):
        batch_list = []
        results = []

        num_threads = 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        def to_cache_device(x: Dict):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    if cache_images_option == 'gpu':
                        x[k] = v.to(self.device)
                    elif cache_images_option == 'cpu':
                        x[k] = v.pin_memory()

        def get_data(idx):
            data = self.mi_train_dataset[idx]
            to_cache_device(data)
            data['image_idx'] = idx
            return data

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in range(len(self.mi_train_dataset)):
                res = executor.submit(get_data, idx)
                results.append(res)

            for res in track(results, description="Caching Mitsuba data batch", transient=True):
                batch_list.append(res.result())

        return batch_list

    def create_mi_train_dataset(self, scale_factor=None) -> TDataset:
        """Sets up the data loaders for training"""
        if scale_factor is None:
            scale_factor = self.config.camera_res_scale_factor
        return self.dataset_type(
            dataparser_outputs=self.mi_train_dataparser_outputs,
            scale_factor=scale_factor,
            pre_mult_mask=self.config.pre_mult_mask,
        )

    def setup_mitsuba_train(self):
        self.cached_train = self.cache_images(self.config.cache_images)
        self.mi_train_sensor_generator = MitsubaSensorGenerator(
            self.mi_train_dataset.cameras.to(self.device),
            self.train_dataparser_outputs.dataparser_scale,  # Explicitly use the original scene_scale
            self.train_camera_optimizer,
        )

    def rescale_train(self, scale_factor):
        CONSOLE.print(f"[bold green]Setting up Mitsuba training dataset of scale {scale_factor}...")
        self.mi_train_dataset = self.create_mi_train_dataset(scale_factor)
        self.setup_mitsuba_train()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        if self.current_record_step != step:
            self.current_record_step = 0
            self.rays_per_step = 0
        self.rays_per_step += super().get_train_rays_per_batch()
        return super().next_train(step)

    def next_train_mitsuba(self, step: int, pose_optimizer: Optional[CameraOptimizer] = None) -> Tuple[mi.Sensor, Dict]:
        if self.current_record_step != step:
            self.current_record_step = 0
            self.rays_per_step = 0
        # self.rays_per_step += self.config.mi_patch_size ** 2 # leave the model to count
        self.train_count += 1

        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [x for x in self.valid_indices]

        batch = self.cached_train[image_idx]
        mi_sensor = self.mi_train_sensor_generator(image_idx)
        return mi_sensor, batch

    def get_train_rays_per_batch(self) -> int:
        return self.rays_per_step

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[MitsubaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is MitsubaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is MitsubaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is MitsubaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default
