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
Code for camera paths.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length
from scipy.spatial.transform.rotation import Rotation as R


def get_interpolated_camera_path(cameras: Cameras, steps: int, order_poses: bool) -> Cameras:
    """Generate a camera path between two cameras. Uses the camera type of the first camera

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps, order_poses=order_poses)

    cameras = Cameras(
        fx=Ks[:, 0, 0],
        fy=Ks[:, 1, 1],
        cx=Ks[0, 0, 2],
        cy=Ks[0, 1, 2],
        camera_type=cameras.camera_type[0],
        camera_to_worlds=poses,
    )
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a spiral trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)

    times = None
    if camera.times is not None:
        times = torch.linspace(0, 1, steps)[:, None]
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
        times=times,
    )


def look_at(origin, target, up):
    """
    Generates a 4x4 transformation matrix that transforms points from a
    coordinate system defined by the origin, target, and up vectors to
    the world coordinate system.

    Args:
        origin (numpy.ndarray): A 1D array of length 3 representing the
            position of the camera in the world coordinate system.
        target (numpy.ndarray): A 1D array of length 3 representing the
            position that the camera is looking at in the world coordinate
            system.
        up (numpy.ndarray): A 1D array of length 3 representing the up
            direction of the camera in the world coordinate system.

    Returns:
        numpy.ndarray: A 4x4 transformation matrix.
    """
    # Compute the forward direction of the camera.
    forward = (target - origin) / np.linalg.norm(target - origin)

    # Compute the right direction of the camera.
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # Compute the up direction of the camera.
    new_up = np.cross(right, forward)

    # Construct a 4x4 transformation matrix.
    transformation = np.eye(4)
    transformation[:3, 0] = right
    transformation[:3, 1] = new_up
    transformation[:3, 2] = -forward
    transformation[:3, 3] = origin

    return transformation


def pose_to_euler_radius(pose):
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    direction = rot @ np.array([0, 0, 1])
    t = -trans[2] / direction[2]
    center = trans + t * direction
    euler_angle = R.from_matrix(rot).as_euler('xyz', degrees=True)
    radius = np.linalg.norm(trans - center)
    return euler_angle, radius, center


def euler_radius_to_pose(euler_angle, radius, center):
    rot = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()
    trans = rot @ np.array([0, 0, radius]) + center
    ret = np.eye(4)
    ret[:3, :3] = rot
    ret[:3, 3] = trans
    return ret


def get_blender_test_path(
        camera: Cameras,
        num_views=360,
        phi_min=45,
        phi_max=135,
        cur_phi_dir=1,
        cur_theta_dir=1,
):
    euler_angle, radius, center = pose_to_euler_radius(camera.camera_to_worlds[0].cpu().numpy())
    theta_step = 360 * 3 / num_views
    euler_angle[0] = min(max(euler_angle[0], phi_min), phi_max)
    # phi_min = min(phi_min, euler_angle[0])
    # phi_max = max(phi_max, euler_angle[0])
    phi_step = 2 * (phi_max - phi_min) / num_views
    c2ws_np = []
    for i in range(num_views):
        c2ws_np.append(euler_radius_to_pose(euler_angle, radius, center)[:3])
        euler_angle[0] += phi_step * cur_phi_dir
        euler_angle[2] += theta_step * cur_theta_dir
        if euler_angle[0] >= phi_max:
            euler_angle[0] = phi_max
            cur_phi_dir *= -1
        elif euler_angle[0] <= phi_min:
            euler_angle[0] = phi_min
            cur_phi_dir *= -1
    c2ws_np = np.stack(c2ws_np)
    c2ws = torch.from_numpy(c2ws_np).float().to(camera.device)
    times = None
    if camera.times is not None:
        times = torch.linspace(0, 1, num_views)[:, None]
    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=c2ws,
        times=times
    )


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    elif camera_path["camera_type"].lower() == "vr180":
        camera_type = CameraType.VR180_L
    else:
        camera_type = CameraType.PERSPECTIVE

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        if camera_type in [
            CameraType.EQUIRECTANGULAR,
            CameraType.OMNIDIRECTIONALSTEREO_L,
            CameraType.OMNIDIRECTIONALSTEREO_R,
            CameraType.VR180_L,
            CameraType.VR180_R,
        ]:
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    # Iff ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )
