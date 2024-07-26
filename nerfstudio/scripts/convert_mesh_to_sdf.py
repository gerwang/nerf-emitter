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
convert_mesh_to_sdf.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import numpy as np
import tyro

import mesh_to_sdf
from nerfstudio.utils.rich_utils import CONSOLE
from util import atleast_4d

mi.set_variant('cuda_ad_rgb')


@dataclass
class ConvertMeshToSdf:
    """Generate data of an outer scene and an inner object."""

    # Name of the input file.
    input_path: Path
    # output file
    output_path: Path
    # sdf resolution
    sdf_res: int = 256

    def main(self) -> None:
        """Main function."""
        sdf = atleast_4d(mesh_to_sdf.create_sdf(str(self.input_path), self.sdf_res, offset=0.5))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mi.VolumeGrid(np.array(sdf)).write(str(self.output_path))
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ConvertMeshToSdf).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ConvertMeshToSdf)  # noqa
