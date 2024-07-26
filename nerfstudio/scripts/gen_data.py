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
gen_data.py
"""
from __future__ import annotations

import os.path
import shutil
from typing import Optional, Literal, List

import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant('cuda_ad_rgb')

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from rich.progress import TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, BarColumn, TextColumn, Progress

from nerfstudio.utils.mi_gl_conversion import get_nerfstudio_matrix
from nerfstudio.utils.mi_util import render_aggregate
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from sensors.spherical_sensor import SphericalCamera  # noqa


@dataclass
class GenerateData:
    """Generate data of an outer scene and an inner object."""

    # Name of the combine file.
    combine_path: Path = Path("scenes/combine.xml")
    # Name of the environment emitter file.
    emitter_file: str = "bathroom"
    # Name of the object file.
    object_file: str = "bunny"
    # output directory
    output_path: Path = Path('data/instant-ngp/{}_in_{}')
    # resolution width
    resx: int = 800
    # resolution height
    resy: int = 800
    # spp
    spp: int = 16384
    # spp per batch
    spp_per_batch: int = 4096
    # number of images
    n_images: int = 100
    # use inside-out instead of outside-in
    inside_out: bool = False
    # camera distance to origin
    distance: float = 0.75
    # x_fov, in degrees
    x_fov: float = 90
    # aabb scle
    aabb_scale: float = 0.6666
    # envmap width
    envmap_resx: int = 1024
    # envmap height
    envmap_resy: int = 512
    # skip environment map lit scene
    skip_env_lit: bool = False
    # skip environment map
    skip_envmap: bool = False
    # mitsuba seed
    seed: Optional[int] = None
    # mitsuba integrator type
    integrator: Literal['path', 'direct'] = 'path'
    # mesh directory path passed to mitsuba xml to load mesh
    mesh_path: Optional[Path] = None
    # output image file format
    image_format: Literal['exr', 'png'] = 'exr'
    # rotations to simulate multi-light settings
    rotations: List[int] = field(default_factory=lambda: [])
    # restrict minimum y coordinate
    minimum_y: Optional[float] = None
    # restrict maximum y coordinate
    maximum_y: Optional[float] = None
    # upside down
    upside_down: bool = False
    # target center
    target: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # whether to perform denoise
    denoise: bool = False
    # use spiral path instead of random path
    spiral: bool = False
    # control the spiral
    start_pitch: float = 5.0
    pitch_delta: float = 1.0
    max_pitch: float = 45.0
    min_pitch: float = 5.0
    n_circle: float = 1.0
    start_angle: float = 0.0
    # mesh type in mesh_path
    mesh_type: Literal['diffuse', 'principled'] = 'principled'
    # change distance when render
    spiral_distance: bool = False
    min_distance: float = 0.5
    max_distance: float = 1.0
    distance_delta: float = 0.0

    def render_outside_in(self, scene, sensor, integrator, to_worlds, output_path, input_masks=None,
                          rotation=None, res=None, last=True):
        image_output_path = output_path / 'images'
        config_output_path = output_path / 'transforms.json'
        image_output_path.mkdir(parents=True, exist_ok=True)
        if res is None:
            res = {
                'frames': [],
            }
        start_idx = len(res['frames'])
        progress = Progress(
            TextColumn(":movie_camera: Rendering :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(
                text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                show_speed=True,
            ),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=False, compact=False),
            TimeElapsedColumn(),
        )

        output_masks = []
        with progress:
            for i in progress.track(range(self.n_images), description="Render outside in"):
                params = mi.traverse(sensor)
                params['to_world'] = to_worlds[start_idx + i]
                params.update()
                with dr.suspend_grad():
                    img = render_aggregate(scene, sensor=sensor, integrator=integrator, spp=self.spp,
                                           spp_per_batch=self.spp_per_batch, denoise=self.denoise)
                img_path = str(image_output_path / f'{start_idx + i:04d}.{self.image_format}')
                if img.shape[-1] == 4:
                    mask = img[..., -1]
                    img = img[..., :-1]
                    output_masks.append(mask)
                elif input_masks is not None:
                    mask = input_masks[start_idx + i]
                else:
                    mask = None
                if self.image_format == 'exr':
                    mi.Bitmap(img).write(img_path)
                else:
                    mi.util.write_bitmap(img_path, img)
                frame = {
                    'file_path': os.path.relpath(img_path, output_path),
                    'transform_matrix': get_nerfstudio_matrix(sensor, scale=True),
                }
                if rotation is not None:
                    frame['rotation'] = rotation
                if mask is not None:
                    mask_output_path = output_path / 'masks'
                    if not mask_output_path.exists():
                        mask_output_path.mkdir(parents=True, exist_ok=True)
                    mask_path = str(mask_output_path / f'{start_idx + i:04d}.png')
                    mi.util.write_bitmap(mask_path, mi.Bitmap(mask))
                    frame['mask_path'] = os.path.relpath(mask_path, output_path)
                res['frames'].append(frame)
        if last:
            res.update({
                'x_fov': self.x_fov,
                'w': self.resx,
                'h': self.resy,
                'aabb_scale': self.aabb_scale,
            })
            json.dump(res, open(config_output_path, 'w'), indent=2)
        return output_masks, res

    def gen_sensor(self, pixel_format='rgb'):
        film = mi.load_dict({
            'type': 'hdrfilm', 'width': self.resx, 'height': self.resy,
            'pixel_format': pixel_format, 'pixel_filter': {'type': 'box' if self.denoise else 'gaussian'},
            'sample_border': True})
        sampler = mi.load_dict({'type': 'independent'})
        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': self.x_fov,
            'sampler': sampler,
            'film': film,
        })
        return sensor

    def gen_integrator(self, hide_emitters=False):
        integrator_dict = {'type': self.integrator, 'hide_emitters': hide_emitters}
        if self.denoise:
            integrator_dict = {
                'type': 'aov',
                'aovs': 'albedo:albedo,normals:sh_normal',
                'integrator': integrator_dict
            }
        integrator = mi.load_dict(integrator_dict)
        return integrator

    def render_envmap(self, scene, rotated_scenes):
        env_output_path = self.output_path / 'env.exr'
        env_output_path.parent.mkdir(parents=True, exist_ok=True)
        sensor = mi.load_dict({
            'type': 'spherical',
            'film': {
                'type': 'hdrfilm',
                'width': self.envmap_resx,
                'height': self.envmap_resy,
                'filter': {
                    'type': 'gaussian',
                    'stddev': 0.25,
                }
            },
            'sampler': {
                'type': 'independent',
            },
            'near_clip': self.distance,
            'to_world': mi.ScalarTransform4f.translate(self.target),
        })
        with dr.suspend_grad():
            img = render_aggregate(scene, sensor=sensor, integrator=self.gen_integrator(), spp=self.spp,
                                   spp_per_batch=self.spp_per_batch, denoise=self.denoise)
        mi.Bitmap(img).write(str(env_output_path))

        for rotation, rotated_scene in zip(self.rotations, rotated_scenes):
            env_output_path = self.output_path / f'env_{rotation}.exr'
            with dr.suspend_grad():
                img = render_aggregate(rotated_scene, sensor=sensor, integrator=self.gen_integrator(), spp=self.spp,
                                       spp_per_batch=self.spp_per_batch, denoise=self.denoise)
            mi.Bitmap(img).write(str(env_output_path))

    def render_env_lit(self, to_worlds, output_path):
        DATASET_ENV_PATH = Path('differentiable-sdf-rendering/assets/emitters/dataset_env.xml')
        env_output_path = self.output_path / 'env.exr'
        combine_parent_path = self.combine_path.parent
        env_scene = mi.load_file(
            str(self.combine_path),
            emitter_file=os.path.relpath(DATASET_ENV_PATH, combine_parent_path),
            envmap_filename=os.path.relpath(env_output_path, combine_parent_path),
            object_file=f'objects/{self.object_file}/shape.xml')
        sensor = self.gen_sensor(pixel_format='rgba')
        integrator = self.gen_integrator(hide_emitters=True)
        masks, res = self.render_outside_in(env_scene, sensor, integrator, to_worlds, output_path,
                                            rotation=0 if len(self.rotations) > 0 else None,
                                            last=len(self.rotations) == 0)
        shutil.copy(env_output_path, output_path / 'env.exr')
        for i, rotation in enumerate(self.rotations):
            env_output_path = self.output_path / f'env_{rotation}.exr'
            env_scene = mi.load_file(
                str(self.combine_path),
                emitter_file=os.path.relpath(DATASET_ENV_PATH, combine_parent_path),
                envmap_filename=os.path.relpath(env_output_path, combine_parent_path),
                object_file=f'objects/{self.object_file}/shape_{rotation}.xml')
            rotated_masks, res = self.render_outside_in(env_scene, sensor, integrator, to_worlds, output_path,
                                                        rotation=rotation, res=res, last=i + 1 == len(self.rotations))
            masks.extend(rotated_masks)

        return masks

    def main(self) -> None:
        """Main function."""
        self.spp_per_batch = min(self.spp_per_batch, self.spp)
        self.output_path = Path(str(self.output_path).format(self.object_file, self.emitter_file))

        mts_args = {
            'emitter_file': f'emitters/{self.emitter_file}/scene.xml',
        }
        if self.mesh_path is not None:
            mts_args.update({
                'mesh_path': self.mesh_path,
                'object_file': {
                    'diffuse': 'differentiable-sdf-rendering/assets/objects/diffuse_mesh.xml',
                    'principled': 'differentiable-sdf-rendering/assets/objects/principled_mesh.xml',
                }[self.mesh_type]
            })
        else:
            mts_args.update({
                'object_file': f'objects/{self.object_file}/shape.xml',
            })

        scene = mi.load_file(str(self.combine_path), **mts_args)
        rotated_scenes = []
        for rotation in self.rotations:
            mts_args.update({
                'object_file': f'objects/{self.object_file}/shape_{rotation}.xml',
            })
            rotated_scene = mi.load_file(str(self.combine_path), **mts_args)
            rotated_scenes.append(rotated_scene)

        to_worlds = []
        sampler = mi.load_dict({'type': 'independent'})
        if self.seed is not None:
            sampler.seed(self.seed, 1)
        pitch = self.start_pitch
        for i in range(self.n_images * (1 + len(self.rotations))):
            while True:
                if self.spiral:
                    d = (mi.Transform4f.rotate(
                        axis=[0, 1, 0], angle=i / self.n_images * 360 * self.n_circle + self.start_angle,
                    ).rotate(
                        axis=[0, 0, 1], angle=pitch,
                    ) @ mi.Vector3f(1., 0., 0.)).numpy()
                    up = np.array([0, 1., 0])
                elif self.upside_down:
                    d = np.array([0, 1., 0])
                    up = np.array([1., 0, 0])
                else:
                    d = mi.warp.square_to_uniform_sphere(sampler.next_2d()).numpy()[0]
                    up = np.array([0, 1., 0])
                d *= self.distance
                target = np.array(self.target)
                origin = d + target
                if self.minimum_y is not None and origin[1] < self.minimum_y:
                    continue
                if self.maximum_y is not None and origin[1] > self.maximum_y:
                    continue
                if self.inside_out:
                    target = 2 * origin - target
                to_world = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)
                to_worlds.append(to_world)
                break
            if self.spiral:
                pitch += self.pitch_delta
                if pitch >= self.max_pitch:
                    pitch = self.max_pitch
                    self.pitch_delta *= -1
                if pitch <= self.min_pitch:
                    pitch = self.min_pitch
                    self.pitch_delta *= -1
            if self.spiral_distance:
                self.distance += self.distance_delta
                if self.distance >= self.max_distance:
                    self.distance = self.max_distance
                    self.distance_delta *= -1
                if self.distance <= self.min_distance:
                    self.distance = self.min_distance
                    self.distance_delta *= -1

        # render envmap
        if not self.skip_envmap:
            CONSOLE.print('Render envmap')
            self.render_envmap(scene, rotated_scenes)
        # render envmap-lit images
        masks = None
        if not self.skip_env_lit:
            CONSOLE.print('Render env lit')
            masks = self.render_env_lit(to_worlds, self.output_path.parent / (self.output_path.name + '_env'))
        # render outside-in images, save nerfstudio cameras
        CONSOLE.print('Render outside in')
        sensor = self.gen_sensor(pixel_format='rgb')
        integrator = self.gen_integrator(hide_emitters=False)
        _, res = self.render_outside_in(scene, sensor, integrator,
                                        to_worlds, self.output_path, masks,
                                        rotation=0 if len(self.rotations) > 0 else None,
                                        last=len(self.rotations) == 0)
        for i, (rotation, rotated_scene) in enumerate(zip(self.rotations, rotated_scenes)):
            _, res = self.render_outside_in(rotated_scene, sensor, integrator,
                                            to_worlds, self.output_path, masks,
                                            rotation=rotation, res=res, last=i + 1 == len(self.rotations))
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(GenerateData).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(GenerateData)  # noqa
