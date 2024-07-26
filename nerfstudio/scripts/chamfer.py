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
chamfer.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
import trimesh.sample
import tyro
from chamferdist import ChamferDistance

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ComputeChamferDistance:
    """
    Compute chamfer distance and output the result to stdout.
    See https://github.com/NVlabs/nvdiffrec/issues/71 for method details.
    """

    input1: Path
    input2: Path
    n_points: int = 2500000
    """Number of points to sample"""
    clip: bool = False
    """Whether to clip mesh"""
    clip_height: float = 0.05
    """Clip the bottom to exclude it from computation"""
    save_mesh: bool = False
    """Save processed mesh"""
    save_point: bool = False
    """Save processed point"""
    save_path: Optional[Path] = None
    """Mesh save path"""
    keep_largest: bool = True
    """Keep largest component only"""
    error_path: Optional[Path] = None
    """Error vector path"""
    single_dir: bool = False
    """Compute single direction"""

    @staticmethod
    def clip_mesh(mesh, clip_height):
        mesh = mesh.slice_plane(np.array([0, clip_height, 0]), np.array([0, 1, 0]))  # y-up
        return ComputeChamferDistance.keep_largest_part(mesh)

    @staticmethod
    def sample_mesh(m, n):
        vpos, _ = trimesh.sample.sample_surface_even(m, n)
        return vpos

    @staticmethod
    def to_cuda(vpos):
        return torch.tensor(vpos, dtype=torch.float32, device="cuda").unsqueeze(0)

    @staticmethod
    def keep_largest_part(mesh: trimesh.Trimesh):
        # only save the connection part with the largest area
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float32)
        mesh = components[areas.argmax()]
        return mesh

    def chamfer_distance_mesh(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh):
        if self.keep_largest:
            if self.error_path is None:
                mesh1 = self.keep_largest_part(mesh1)
            mesh2 = self.keep_largest_part(mesh2)
        if self.clip:
            if self.error_path is None:
                mesh1 = self.clip_mesh(mesh1, self.clip_height)
            mesh2 = self.clip_mesh(mesh2, self.clip_height)
            if self.save_mesh:
                self.save_path.mkdir(exist_ok=True, parents=True)
                mesh1.export(self.save_path / f'{self.input1.name}.ply')
                mesh2.export(self.save_path / f'{self.input2.name}.ply')
        if self.error_path is None:
            vert1 = self.sample_mesh(mesh1, self.n_points)
        else:
            vert1 = mesh1.vertices
        vert2 = self.sample_mesh(mesh2, self.n_points)
        if self.save_point:
            self.save_path.mkdir(exist_ok=True, parents=True)
            trimesh.PointCloud(vert1).export(self.save_path / f'{self.input1.name}_point.ply')
            trimesh.PointCloud(vert2).export(self.save_path / f'{self.input2.name}_point.ply')
        vert1 = self.to_cuda(vert1)
        vert2 = self.to_cuda(vert2)
        chamfer_dist = ChamferDistance()
        with torch.no_grad():
            if self.error_path is None:
                dist_forward = chamfer_dist(vert1, vert2, bidirectional=not self.single_dir,
                                            batch_reduction=None, point_reduction='mean')[0]
            else:
                dist_forward = chamfer_dist(vert1, vert2, bidirectional=False,
                                            batch_reduction=None, point_reduction=None)[0]
                if not self.error_path.parent.exists():
                    self.error_path.parent.mkdir(parents=True)
                np.save(self.error_path, dist_forward.cpu().numpy())
                dist_forward = dist_forward.mean()
            dist_forward = float(dist_forward)
        return dist_forward

    def chamfer_distance_path(self, input1: Path, input2: Path):
        mesh1 = trimesh.load(input1)
        mesh2 = trimesh.load(input2)
        return self.chamfer_distance_mesh(mesh1, mesh2)

    def main(self) -> None:
        """Main function."""
        dist = self.chamfer_distance_path(self.input1, self.input2)
        CONSOLE.print(dist)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeChamferDistance).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeChamferDistance)  # noqa
