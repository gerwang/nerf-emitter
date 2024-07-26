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

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import cv2.aruco as aruco
import numpy as np
import tyro
from tqdm import tqdm

from nerfstudio.process_data.process_data_utils import list_images
from nerfstudio.utils.rich_utils import CONSOLE


def gen_board():
    # Define parameters for the CharucoBoard
    num_squares_x = 7
    num_squares_y = 10
    square_length = 0.04  # length of each square side in meters
    marker_length = 0.02  # length of the markers in meters
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)  # you can choose a different dictionary

    # Define a nonzero start ID for aruco markers
    start_id = 200

    # Create CharucoBoard with a nonzero start ID
    board1 = aruco.CharucoBoard(
        (num_squares_x, num_squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=dictionary,
        ids=np.arange(start_id, start_id + num_squares_x * num_squares_y // 2, dtype=np.int32)
    )

    board2 = aruco.CharucoBoard(
        (num_squares_x, num_squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=dictionary,
        ids=board1.getIds() + len(board1.getIds()),
    )
    return board1, board2, dictionary


def get_marker_info(calib_dir, dictionary, board, crop_ratio=1, filter_list=None):
    file_lst = list_images(Path(calib_dir))
    if filter_list is not None:
        file_lst = [x for x in file_lst if x.stem in filter_list]
    progress_bar = tqdm(enumerate(file_lst), total=len(file_lst), unit="%",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]")

    res_list = []
    res_ids_lst = []
    img_name_lst = []

    use_crop = False
    if crop_ratio != 1:
        use_crop = True

    for i, file_path in progress_bar:
        img_name = file_path.stem
        img = cv2.imread(str(file_path))
        if use_crop:
            height, width = img.shape[:2]
            img = img[int(height * (1 - crop_ratio) / 2):int(height * (1 + crop_ratio) / 2),
                  int(width * (1 - crop_ratio) / 2):int(width * (1 + crop_ratio) / 2)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markerCorners1, markerIds1, rejectedImgPoints1 = cv2.aruco.detectMarkers(img, dictionary)
        if len(markerCorners1) == 0:
            continue
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners1, markerIds1, gray, board)
        if retval == 0:
            continue

        # because we crop the image, we need to add the offset back
        if use_crop:
            charuco_corners[:, 0, :] += np.array([int(width * (1 - crop_ratio) / 2), int(height * (1 - crop_ratio) / 2)])

        res_list.append(charuco_corners[:, 0, :])
        res_ids_lst.append(charuco_ids[:, 0])
        img_name_lst.append(img_name)

        progress_percent = (i + 1) / len(file_lst) * 100

        # 更新进度条
        progress_bar.set_postfix(progress="{:.2f}%".format(progress_percent))
        progress_bar.update(1)

    # 创建一个字典，用于保存 marker 信息，格式为
    points_dict = {}
    for i, img_name in enumerate(img_name_lst):
        for j, marker_id in enumerate(res_ids_lst[i]):
            if str(marker_id) not in points_dict:
                points_dict[str(marker_id)] = []
            points_dict[str(marker_id)].append({
                "camera_id": str(img_name),
                "x": str(res_list[i][j][0]),
                "y": str(res_list[i][j][1])
            })

    return points_dict


def extract_camera_relationship(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameras = []
    for camera_elem in root.iter('camera'):
        camera_id = camera_elem.get('id')
        label = camera_elem.get('label')
        camera_info = {'id': camera_id, 'label': label}
        cameras.append(camera_info)

    return cameras


def replace_camera_id(dictionary, cameras):
    new_dict = {}
    id_mapping = {}
    for camera in cameras:
        camera_id = camera.get('id')
        label = camera.get('label')
        id_mapping[label] = camera_id

    for key, values in dictionary.items():
        new_values = []
        for value in values:
            new_value = {}
            camera_id = value.get('camera_id')
            if camera_id in id_mapping:
                new_value['camera_id'] = id_mapping[camera_id]
                new_value['x'] = value.get('x')
                new_value['y'] = value.get('y')
                new_values.append(new_value)
        new_dict[key] = new_values

    return new_dict


def grouping(root, id_dict1, id_dict2):
    # 获取所有的 marker 元素
    chunk_element = root.find('.//chunk')

    # 找到 <marker> 元素
    marker_element = chunk_element.find('.//markers')

    # 根据 dict1 和 dict2 key 总长， 计算marker的总数
    marker_count = len(id_dict1) + len(id_dict2)

    # 删除原有的 marker 元素
    for marker in marker_element.findall('marker'):
        marker_element.remove(marker)

    marker_element.set('next_id', str(marker_count))
    marker_element.set('next_group_id', '2')

    group1_element = ET.SubElement(marker_element, 'group')
    group1_element.set('id', '0')
    group1_element.set('label', 'board1')

    group2_element = ET.SubElement(marker_element, 'group')
    group2_element.set('id', '1')
    group2_element.set('label', 'board2')

    for new_id, label in id_dict1.items():
        new_marker = ET.SubElement(group1_element, 'marker')
        new_marker.set('id', str(new_id))
        new_marker.set('label', "b1_" + str(int(new_id) - 1))
        new_marker.tail = "\n        "

    for new_id, label in id_dict2.items():
        new_marker = ET.SubElement(group2_element, 'marker')
        new_marker.set('id', str(new_id))
        new_marker.set('label', "b2_" + str(int(new_id) - 55))
        new_marker.tail = "\n        "


def rename_dict_keys_by_add(dictionary, increment):
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(key, str) and key.isdigit():
            new_key = int(key) + increment
            new_dict[str(new_key)] = value
        else:
            exit()
    return new_dict


def rename_dict_keys_by_enumerating(dictionary):
    renamed_dict = {}
    n = len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        new_key = str(i)
        renamed_dict[new_key] = dictionary[key]
    return renamed_dict


def main(xml_file, board1, board2, dictionary, calib_dir, output_file, crop_ratio):
    # find all xml files in xml_dir
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameras = extract_camera_relationship(xml_file)
    known_info_dict = get_marker_info(calib_dir, dictionary, board1, crop_ratio,
                                      filter_list=[x['label'] for x in cameras])
    known_info_dict = replace_camera_id(known_info_dict, cameras)

    cameras2 = extract_camera_relationship(xml_file)
    known_info_dict2 = get_marker_info(calib_dir, dictionary, board2, crop_ratio,
                                       filter_list=[x['label'] for x in cameras2])
    known_info_dict2 = replace_camera_id(known_info_dict2, cameras2)

    # dict1_len = len(known_info_dict)
    # known_info_dict = rename_dict_keys_by_enumerating(known_info_dict)
    # known_info_dict2 = rename_dict_keys_by_enumerating(known_info_dict2)
    known_info_dict = rename_dict_keys_by_add(known_info_dict, 1)
    known_info_dict2 = rename_dict_keys_by_add(known_info_dict2, 55)

    # assert 两个dict key不重复
    for key in known_info_dict.keys():
        assert key not in known_info_dict2.keys()

    grouping(root, known_info_dict, known_info_dict2)

    # 找到 <frame id="0"> 元素
    frame_elem = root.find('.//frame[@id="0"]')

    # 清空原有的 marker 信息
    markers_elem = frame_elem.find('markers')
    if markers_elem is not None:
        frame_elem.remove(markers_elem)

    # 添加新的 marker 信息
    # 添加新的 marker 信息
    markers_elem = ET.SubElement(frame_elem, 'markers')
    for marker_id, camera_list in known_info_dict.items():
        marker_elem = ET.SubElement(markers_elem, 'marker')
        marker_elem.set('marker_id', marker_id)

        for camera_info in camera_list:
            location_elem = ET.SubElement(marker_elem, 'location')
            location_elem.set('camera_id', camera_info['camera_id'])
            location_elem.set('pinned', 'true')
            location_elem.set('x', str(camera_info['x']))
            location_elem.set('y', str(camera_info['y']))

            # 使用回车换行增加代码可读性
            location_elem.tail = "\n        "

        marker_elem.tail = "\n      "

    for marker_id, camera_list in known_info_dict2.items():
        marker_elem = ET.SubElement(markers_elem, 'marker')
        marker_elem.set('marker_id', marker_id)

        for camera_info in camera_list:
            location_elem = ET.SubElement(marker_elem, 'location')
            location_elem.set('camera_id', camera_info['camera_id'])
            location_elem.set('pinned', 'true')
            location_elem.set('x', str(camera_info['x']))
            location_elem.set('y', str(camera_info['y']))

            # 使用回车换行增加代码可读性
            location_elem.tail = "\n        "

        marker_elem.tail = "\n      "

    # 使用回车换行增加代码可读性
    markers_elem.tail = "\n    "

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tree.write(f"{output_file}", encoding='utf-8', xml_declaration=True)


@dataclass
class MarkerToXml:
    """Generate data of an outer scene and an inner object."""

    # Template name of the input file.
    xml_path: Path
    # Name of the photo file.
    photo_path: Path
    # output file
    output_path: Path
    # List of degrees
    degree_list: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    # crop ratio
    crop_ratio: float = 1.0

    def main(self) -> None:
        """Main function."""
        board1, board2, dictionary = gen_board()
        for degree in self.degree_list:
            xml_file = str(self.xml_path).format(degree)
            photo_file = str(self.photo_path).format(degree)
            output_file = str(self.output_path).format(degree)
            main(xml_file, board1, board2, dictionary, photo_file, output_file, self.crop_ratio)
            CONSOLE.print(f"Saved results to: {output_file}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MarkerToXml).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(MarkerToXml)  # noqa
