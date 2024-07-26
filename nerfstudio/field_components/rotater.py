from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, Dict

import mitsuba as mi
import torch.nn
from jaxtyping import Float, Int, Bool
from torch import Tensor

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.mi_gl_conversion import torch_gl2mi_scale_shifted_matrix, batch_affine_left


class RotationBoundType(Enum):
    BOX = auto()
    SPHERE = auto()


class Rotater(torch.nn.Module):
    def __init__(self, rotations: list[float], scene_scale: float,
                 transform_matrices: Optional[Dict[float, Tensor[Float, "n_rotation 4 4"]]] = None,
                 rotation_aabb: Optional[Tensor[Float, "2 3"]] = None):
        super().__init__()
        mi.set_variant('cuda_ad_rgb')
        # gl mi conversion
        self.register_buffer('torch_gl2mi', torch_gl2mi_scale_shifted_matrix(scene_scale))
        self.register_buffer('torch_mi2gl', torch.linalg.inv(self.torch_gl2mi))
        # determine which rotation to take
        self.unique_rotations = sorted(set(rotations))
        self.id_mapping = {k: v for k, v in zip(self.unique_rotations, range(len(self.unique_rotations)))}
        self.rotation_ids = torch.tensor([self.id_mapping[x] for x in rotations])
        self.same_rotations = []
        for i in range(len(self.unique_rotations)):
            self.same_rotations.append(torch.where(self.rotation_ids == i)[0].cpu())
        # determine rotation transforms
        if transform_matrices is not None:
            torch_transforms = []
            for unique_rotation in self.unique_rotations:
                torch_transforms.append(transform_matrices[unique_rotation])
            torch_transforms = torch.stack(torch_transforms, dim=0)
            self.register_buffer('torch_transforms', torch_transforms)
        else:
            if len(rotations) == 3:
                mi_axis = mi.Vector3f(1, 0, 0)
                mi_angles = -mi.Float(self.unique_rotations)
            else:
                mi_axis = mi.Vector3f(0, 1, 0)
                mi_angles = mi.Float(self.unique_rotations)
            mi_center = mi.Point3f(0.5, 0.5, 0.5)
            mi_transforms = mi.Transform4f(mi.Transform4f.translate(mi_center)
                                           .rotate(axis=mi_axis, angle=mi_angles)
                                           .translate(-mi_center))
            self.register_buffer('torch_transforms',
                                 self.torch_mi2gl @ mi_transforms.matrix.torch().cpu() @ self.torch_gl2mi)
        # determine the rotation region
        if rotation_aabb is not None:
            self.rotation_bound_type = RotationBoundType.BOX
            self.register_buffer('rotation_aabb', rotation_aabb)
        else:
            # Use bounding sphere rotation
            self.rotation_bound_type = RotationBoundType.SPHERE
            self.register_buffer('center', batch_affine_left(self.torch_mi2gl, mi_center.torch().cpu()))
            self.register_buffer('radius', torch.tensor(math.sqrt(3) * scene_scale))
        self.enabled = True
        self.rotation_optimizer = None

    def set_rotation_optimizer(self, rotation_optimizer: CameraOptimizer):
        self.rotation_optimizer = rotation_optimizer

    def get_torch_transforms(self, rotation_ids: Int[Tensor, "*batch"] | int):
        torch_transforms = self.torch_transforms
        if self.rotation_optimizer is not None:
            c_tensor = torch.arange(len(torch_transforms), device=torch_transforms.device)
            camera_opt_to_camera = self.rotation_optimizer(c_tensor)
            torch_transforms = pose_utils.multiply(torch_transforms, camera_opt_to_camera)
            torch_transforms = pose_utils.to4x4(torch_transforms).contiguous()
        rotation_transforms = torch_transforms[rotation_ids]
        return rotation_transforms

    def get_mi_transform(self, rotation_id: int):
        with torch.no_grad():
            torch_transform = self.get_torch_transforms(rotation_id)
        mi_transform = mi.Transform4f((self.torch_gl2mi @ torch_transform @ self.torch_mi2gl).cpu().numpy())
        return mi_transform

    def num_rotations(self):
        return len(self.unique_rotations)

    def map_rotation_ids(self, camera_indices):
        if isinstance(camera_indices, torch.Tensor):
            self.rotation_ids = self.rotation_ids.to(camera_indices.device)
            camera_indices = camera_indices.clamp(max=len(self.rotation_ids) - 1)
        else:
            camera_indices = min(camera_indices, len(self.rotation_ids) - 1)
        return self.rotation_ids[camera_indices]

    def apply_mi_sensor(self, sensor, camera_idx: int):
        if not self.enabled:
            return
        rotation_idx = int(self.map_rotation_ids(camera_idx))
        params = mi.traverse(sensor)
        params['to_world'] = self.get_mi_transform(rotation_idx) @ params['to_world']
        params.update()

    def apply_c2w_homo(self, c2w: Float[Tensor, "N 4 4"], camera_indices: Int[Tensor, "N 1"]):
        transforms: Float[Tensor, "N 4 4"] = self.get_torch_transforms(self.map_rotation_ids(camera_indices[..., 0]))
        c2w = transforms @ c2w
        return c2w

    def apply_sdf_scene(self, sdf_scene, camera_idx: int):
        if not self.enabled:
            return
        rotation_idx = int(self.map_rotation_ids(camera_idx))
        params = mi.traverse(sdf_scene.environment())
        params['to_world'] = self.get_mi_transform(rotation_idx)
        params.update()
        if hasattr(sdf_scene.environment(), 'set_camera_idx'):
            sdf_scene.environment().set_camera_idx(camera_idx)

    def get_rotation_mask(self, positions: Tensor[Float, "... 3"]) -> Tensor[Bool, "... 1"]:
        if self.rotation_bound_type == RotationBoundType.SPHERE:
            return torch.linalg.norm(positions - self.center, dim=-1, keepdim=True) < self.radius
        else:
            return ((self.rotation_aabb[0] <= positions).all(dim=-1, keepdim=True) &
                    (positions <= self.rotation_aabb[1]).all(dim=-1, keepdim=True))

    def apply_frustums(self, frustums: Frustums, camera_indices: torch.Tensor):
        if not self.enabled:
            return
        positions = frustums.get_positions()
        rotate_mask = self.get_rotation_mask(positions)
        transforms = self.get_torch_transforms(self.map_rotation_ids(camera_indices[..., 0]))
        frustums.origins = torch.where(
            rotate_mask,
            batch_affine_left(
                transforms, frustums.origins[..., :1, :]
            ).expand(frustums.origins.shape),
            frustums.origins)
        frustums.directions = torch.where(
            rotate_mask,
            batch_affine_left(
                transforms, frustums.directions[..., :1, :], is_pos=False
            ).expand(frustums.directions.shape),
            frustums.directions)

    def get_rotation_transforms(self, camera_indices):
        rotation_ids = self.map_rotation_ids(camera_indices)
        return self.get_torch_transforms(rotation_ids)

    def same_rotation_index_groups(self) -> list[Int[Tensor, "b"]]:
        return self.same_rotations

    def get_rotation_options(self):
        return [str(x) for x in self.unique_rotations]

    def map_rotation_option_to_camera_idx(self, rotation_option: str) -> int:
        rotation_idx = float(rotation_option)
        return int(self.same_rotations[self.id_mapping[rotation_idx]][0])

    def apply_scene_box(self, scene_box: SceneBox, camera_indices: Int[Tensor, "*batch 1"]):
        if not self.enabled:
            return
        scene_box.from_world = self.get_rotation_transforms(camera_indices[..., 0])
