"""
Run with blender --background --factory-startup inner_outer_box.blend --python ~/mnt-183/git-task/ir_repos/nerfstudio-ir/nerfstudio/scripts/inner_outer_box.py

"""
import os

import bpy
import numpy as np

scene = bpy.context.scene

inner_box = scene.objects['InnerBox']
outer_box = scene.objects['OuterBox']
empty = scene.objects['Empty']

empty.location = (0, 0, 0)
empty.rotation_euler = (0, 0, 0)
empty.scale = (1, 1, 1)

bpy.context.view_layer.update()

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


inv_inner_box_transform = np.linalg.inv(np.array(listify_matrix(inner_box.matrix_world)))
outer_box_transform = np.array(listify_matrix(outer_box.matrix_world))

blend_file_path = os.path.dirname(bpy.data.filepath)
cube_vertices = np.array(
    [[-1, -1, -1],
     [-1, -1, 1],
     [-1, 1, -1],
     [-1, 1, 1],
     [1, -1, -1],
     [1, -1, 1],
     [1, 1, -1],
     [1, 1, 1]])


def to_homo(points_3d):
    """
    Convert Nx3 array to Nx4 homogeneous coordinates.

    Parameters:
    - points_3d: Nx3 NumPy array representing 3D points.

    Returns:
    - Nx4 NumPy array representing homogeneous coordinates.
    """
    # Add a column with the value 1 to make it homogeneous
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    return points_homogeneous


cube_vertices = (inv_inner_box_transform @ outer_box_transform @ to_homo(cube_vertices).T).T[..., :3]
aabb = np.stack([
    cube_vertices.min(axis=0),
    cube_vertices.max(axis=0)
], axis=0)

np.savetxt(f'{blend_file_path}/outer_box_aabb.txt', aabb)
np.savetxt(f'{blend_file_path}/inv_inner_box_transform.txt', inv_inner_box_transform)
