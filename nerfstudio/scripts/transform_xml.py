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
transform_xml.py
Apply an affine transform to a Mitsuba XML file
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mitsuba as mi
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from nerfstudio.utils.rich_utils import CONSOLE

mi.set_variant('scalar_rgb')


class CommentedTreeBuilder(ET.TreeBuilder):
    def comment(self, data):
        self.start(ET.Comment, {})
        self.data(data)
        self.end(ET.Comment)


def exclude_scale_component(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]
    # Calculate the rotation as a quaternion
    rotation = Rotation.from_matrix(rotation_matrix)
    rotation_quaternion = rotation.as_quat()
    # Create a new transformation matrix with only rotation and translation
    new_matrix = np.eye(4)
    new_matrix[:3, :3] = Rotation.from_quat(rotation_quaternion).as_matrix()
    new_matrix[:3, 3] = translation_vector
    return new_matrix


def transform_nodes(transform, nodes):
    # Do something with the 'transform' nodes found
    for node in nodes:
        # Do something with the 'transform' node
        matrices = node.findall(".//matrix")
        for matrix in matrices:
            orig_transform = mi.Transform4f(np.fromstring(matrix.attrib['value'], sep=' ').reshape(4, 4))
            new_transform = transform @ orig_transform
            matrix.attrib['value'] = ' '.join(new_transform.matrix.numpy().flatten().astype(str))


@dataclass
class TransformXml:
    """Generate data of an outer scene and an inner object."""

    # Name of the input file.
    input_path: Path
    # output file
    output_path: Path
    # transform str in drjit
    transform: str = 'mi.Transform4f()'

    def main(self) -> None:
        """Main function."""
        transform_dr = eval(self.transform, {'mi': mi})
        transform_np = np.array(transform_dr.matrix)
        new_matrix = exclude_scale_component(transform_np)
        transform_cam = mi.Transform4f(new_matrix)

        parser = ET.XMLParser(target=CommentedTreeBuilder())
        tree = ET.parse(self.input_path, parser=parser)
        root = tree.getroot()

        # find all nodes tagged 'shape' without a child node named 'transform'
        shapes = root.findall(".//shape[@type!='shapegroup']")

        # for each 'shape' node found, add a 'transform' child node with a 'matrix' child node
        for shape in shapes:
            if shape.find('transform') is not None:
                continue
            transform = ET.Element('transform', {'name': 'to_world'})
            ET.SubElement(transform, 'matrix', {'value': '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'})
            shape.append(transform)

        # find all nodes tagged 'transform' with a name of 'to_world'
        transforms = root.findall(".//transform[@name='to_world']")

        # for each 'transform' node found, find its 'matrix' child nodes and print their values
        transform_nodes(transform_dr, transforms)

        # Find all 'sensor' nodes
        sensors = root.findall('.//sensor')

        transform_cam_delta = transform_cam @ transform_dr.inverse()
        # Iterate over the 'sensor' nodes and find their 'transform' children
        for sensor in sensors:
            transforms = sensor.findall('.//transform')
            transform_nodes(transform_cam_delta, transforms)

        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True)
        # write the modified XML to a new file named 'output.xml'
        tree.write(self.output_path, encoding='utf-8', xml_declaration=False)
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(TransformXml).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TransformXml)  # noqa
