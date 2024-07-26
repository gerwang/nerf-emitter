from __future__ import annotations

import os
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import imageio.v3 as iio
import mediapy as media
import numpy as np
import torch
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import CropMode
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, comms
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn


def indices_by_rank(n_images):
    base_n, extra_n = divmod(n_images, comms.get_world_size())
    prefix_sum = base_n * comms.get_local_rank()
    i = comms.get_local_rank()
    if i < extra_n:
        this_n = base_n + 1
        prefix_sum += i
    else:
        prefix_sum += extra_n
        this_n = base_n
    return list(range(prefix_sum, prefix_sum + this_n))


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


def render_trajectory_video(
        pipeline: Pipeline,
        cameras: Cameras | list,
        output_filename: Path,
        rendered_output_names: List[str],
        crop_data: Optional[CropData] = None,
        rendered_resolution_scaling_factor: float = 1.0,
        seconds: float = 5.0,
        output_format: Literal["images", "video"] = "video",
        image_format: Literal["jpeg", "png", 'exr', 'mask_png'] = "jpeg",
        jpeg_quality: int = 100,
        depth_near_plane: Optional[float] = None,
        depth_far_plane: Optional[float] = None,
        colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
        axis_angles: Optional[List[Tuple[List[float], float]]] = None,
        image_names: Optional[List[str]] = None,
        main_process_only: bool = True,
        quiet: bool = False,
        target_device: torch.device = torch.device('cuda:0'),
        return_output_names: Optional[List[str]] = None,
        camera_indices: Optional[List[int]] = None,
        stack_dim: Literal['width', 'channel'] = 'width',
        occlusion_image_paths: Optional[List[Path]] = None,
        background_image_paths: Optional[List[Path]] = None,
) -> List[Tensor]:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        stack_dim: stack images at channel dimension or width dimension
    """
    return_output_names = [] if return_output_names is None else return_output_names

    def is_main_process():
        return not main_process_only or comms.is_main_process()

    if is_main_process() and not quiet:
        CONSOLE.print("[bold green]Creating trajectory " + output_format)
    if isinstance(cameras, Cameras):
        cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
        cameras = cameras.to(target_device)
    fps = len(cameras) / seconds

    if is_main_process() and not quiet:
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
    else:
        class DummyProcess:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            @staticmethod
            def track(iterable, *args, **kwargs):
                return iterable

        progress = DummyProcess()

    if is_main_process():
        output_image_dir = output_filename.parent / output_filename.stem
        if output_format == "images":
            output_image_dir.mkdir(parents=True, exist_ok=True)
        if output_format == "video":
            # make the folder if it doesn't exist
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            # NOTE:
            # we could use ffmpeg_args "-movflags faststart" for progressive download,
            # which would force moov atom into known position before mdat,
            # but then we would have to move all of mdat to insert metadata atom
            # (unless we reserve enough space to overwrite with our uuid tag,
            # but we don't know how big the video file will be, so it's not certain!)

    res = []
    if stack_dim == 'width':
        stack_axis = 1
    elif stack_dim == 'channel':
        stack_axis = 2
    else:
        raise ValueError(f'unknown {stack_dim}')

    with ExitStack() as stack:
        writer = None

        with progress:
            if camera_indices is None:
                camera_indices = list(range(len(cameras)))
            for camera_idx in progress.track(camera_indices, description=""):
                if is_main_process():
                    if axis_angles is not None:
                        axis, angle = axis_angles[camera_idx]
                        pipeline.set_light_axis_angle(axis, angle)
                outputs = pipeline.render_camera_outputs(cameras, camera_idx, crop_data)
                if occlusion_image_paths is not None and background_image_paths is not None:
                    import mitsuba as mi
                    img, mask = outputs['rgb'], outputs['mask']
                    mask = (mask + 0.01).clamp(0., 1.)
                    occlusion_image = torch.from_numpy(
                        np.asarray(mi.Bitmap(str(occlusion_image_paths[camera_idx])))).float().to(img.device)
                    background_image = torch.from_numpy(
                        np.asarray(mi.Bitmap(str(background_image_paths[camera_idx])))).float().to(img.device)
                    occlusion = occlusion_image[..., :3]
                    occlusion_mask = occlusion_image[..., 3:]
                    background = background_image[..., :3]
                    img = img + (1 - mask) * background
                    img = occlusion + (1 - occlusion_mask) * img
                    outputs['rgb'] = img
                if is_main_process():
                    render_image = []
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs",
                                          justify="center")
                            CONSOLE.print(
                                f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                            )
                            sys.exit(1)
                        output_image = outputs[rendered_output_name]

                        is_depth = rendered_output_name.find("depth") != -1
                        if is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=depth_near_plane,
                                    far_plane=depth_far_plane,
                                    colormap_options=colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        render_image.append(output_image)
                    for return_output_name in return_output_names:
                        if return_output_name not in outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(f"Could not find {return_output_name} in the model outputs",
                                          justify="center")
                            CONSOLE.print(
                                f"Please set --return_output_name to one of: {outputs.keys()}", justify="center"
                            )
                            sys.exit(1)
                        output_image = outputs[return_output_name].detach()
                        res.append(output_image)
                    render_image = np.concatenate(render_image, axis=stack_axis)
                    if image_names:
                        image_name = image_names[camera_idx]
                    else:
                        image_name = f"{camera_idx:05d}"
                    if output_format == "images":
                        if image_format == "png":
                            media.write_image(output_image_dir / f"{image_name}.png", render_image, fmt="png")
                        if image_format == "jpeg":
                            media.write_image(
                                output_image_dir / f"{image_name}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                            )
                        if image_format == 'exr':
                            iio.imwrite(output_image_dir / f"{image_name}.exr", render_image)
                        if image_format == 'mask_png':
                            iio.imwrite(output_image_dir / f"{image_name}.png",
                                        (render_image * np.iinfo(np.uint16).max).astype(np.uint16)[..., 0])
                    if output_format == "video":
                        if writer is None:
                            render_width = int(render_image.shape[1])
                            render_height = int(render_image.shape[0])
                            writer = stack.enter_context(
                                media.VideoWriter(
                                    path=output_filename,
                                    shape=(render_height, render_width),
                                    fps=fps,
                                )
                            )
                        writer.add_image(render_image)

    if is_main_process() and not quiet:
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        if output_format == "video":
            if isinstance(cameras, Cameras) and cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
                CONSOLE.print("Adding spherical camera data")
                insert_spherical_metadata_into_file(output_filename)
            table.add_row("Video", str(output_filename))
        else:
            table.add_row("Images", str(output_image_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

    return res


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    center: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """center of the crop"""
    scale: Float[Tensor, "3"] = torch.Tensor([2.0, 2.0, 2.0])
    """scale of the crop"""
    crop_mode: CropMode = CropMode.NORMAL
    """Which segment to retain, before aabb, inside aabb or beyond aabb"""
