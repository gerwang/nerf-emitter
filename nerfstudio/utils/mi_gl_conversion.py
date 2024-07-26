import numpy as np
import torch

mi2gl_left = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

gl2mi_left = np.linalg.inv(mi2gl_left)

gl2mi_right = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

mi2gl_right = np.linalg.inv(gl2mi_right)


def torch_scale_shifted_gl2mi(c2w: torch.Tensor, scene_scale: float):  # [-scale, scale] to [0, 1]
    res = c2w.clone()
    res[..., :3, 3] = c2w[..., :3, 3] / scene_scale * 0.5 + 0.5
    return res


def torch_scale_shifted_mi2gl(c2w: torch.Tensor, scene_scale: float):  # [-scale, scale] to [0, 1]
    res = c2w.clone()
    res[..., :3, 3] = (c2w[..., :3, 3] * 2 - 1) * scene_scale
    return res


def torch_point_scale_shifted_gl2mi(o: torch.Tensor, scene_scale: float):  # [-scale, scale] to [0, 1]
    o = o / scene_scale * 0.5 + 0.5
    return o


def torch_point_scale_shifted_mi2gl(o: torch.Tensor, scene_scale: float):  # [0, 1] to [-scale, scale]
    o = (o * 2 - 1) * scene_scale
    return o


def get_nerfstudio_matrix(sensor, scale=False, scene_scale=1.0):
    res = np.array(sensor.world_transform().matrix)[0]
    if scale:
        res[:3, 3] = (res[:3, 3] * 2 - 1) * scene_scale
    res = mi2gl_left @ res @ mi2gl_right
    res = res.tolist()
    return res


def torch_gl2mi_scale_shifted_matrix(scene_scale=1.0):
    torch_gl2mi_left = torch.from_numpy(gl2mi_left).float()
    torch_gl2mi_left[:3, :3] *= 0.5 / scene_scale
    torch_gl2mi_left[:3, 3] = 0.5
    return torch_gl2mi_left.unsqueeze(0)


def batch_affine_left(M, x, is_pos=True):
    res = (M[..., :3, :3] @ x.unsqueeze(-1))[..., 0]
    if is_pos:
        res += M[..., :3, 3]
    return res
