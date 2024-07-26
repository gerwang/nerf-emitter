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


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tyro
from torchmetrics.image import PeakSignalNoiseRatio


@dataclass
class MaskedPSNR:
    input_dir: Path
    mask_dir: Path

    def main(self) -> None:

        avg_psnr = 0
        cnt = 0
        psnr_module = PeakSignalNoiseRatio(data_range=1.0)

        for filename in os.listdir(self.input_dir):
            if filename.endswith('.png') and not filename.endswith('-before_occlusion_rgba.png'):
                basename = filename[:-len('-img.png')]
                img = iio.imread(self.input_dir / f'{basename}-img.png')
                img = img / np.iinfo(img.dtype).max
                mask = iio.imread(self.mask_dir / f'{basename}-before_occlusion_rgba.png')[..., 3]
                mask = mask / np.iinfo(mask.dtype).max
                half_width = img.shape[1] // 2
                target, recon = img[:, :half_width], img[:, half_width:]
                binary_mask = mask > 0.5
                target = torch.from_numpy(target).float()
                recon = torch.from_numpy(recon).float()
                binary_mask = torch.from_numpy(binary_mask).bool()
                # print(int(binary_mask.sum()), binary_mask.numel())
                psnr = psnr_module(target[binary_mask], recon[binary_mask])
                # psnr = psnr_module(target, recon)
                avg_psnr += float(psnr)
                cnt += 1

        avg_psnr /= cnt
        print(avg_psnr)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[MaskedPSNR]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(MaskedPSNR)  # noqa
