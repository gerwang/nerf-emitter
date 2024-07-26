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

import os

import numpy as np
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class CompositeImage:
    foreground: Path
    background: Path
    mask: Path
    output: Path

    def main(self) -> None:
        """Main function."""
        self.output.mkdir(parents=True, exist_ok=True)
        filenames = os.listdir(self.foreground)
        for filename in tqdm(filenames):
            if filename.endswith('.exr'):
                foreground_img = np.asarray(mi.Bitmap(str(self.foreground / filename)))
                background_img = np.asarray(mi.Bitmap(str(self.background / filename)))
                mask_img = np.asarray(mi.Bitmap(str(self.mask / filename.replace('.exr', '.png')))).astype(
                    np.float32) / 255.0
                img = foreground_img * mask_img + background_img * (1 - mask_img)
                output_img_path = self.output / filename
                mi.Bitmap(img).write(str(output_img_path))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(CompositeImage).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(CompositeImage)  # noqa
