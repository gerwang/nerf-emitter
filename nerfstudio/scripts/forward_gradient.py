from __future__ import annotations

import os

import mitsuba as mi

mi.set_variant('cuda_ad_rgb')  # TODO set variant while respecting LLVM variant
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.configs.base_config import MachineConfig
from nerfstudio.scripts.train import launch, _set_random_seed
from nerfstudio.utils import comms
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import tyro
import yaml

from nerfstudio.engine.trainer import TrainerConfig
import imageio.v3 as iio


def main(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    _, pipeline, _, _ = eval_setup(
        config_path=None, _config=config,
        local_rank=local_rank, world_size=world_size,
        test_mode='test'
    )

    assert hasattr(config, 'forward_gradient')
    forward_gradient: ForwardGradient = getattr(config, 'forward_gradient')
    forward_gradient.render_forward_gradient(pipeline)


@dataclass
class ForwardGradient:
    """Render forward gradient image."""

    load_config: Path
    """Path to config YAML file."""
    machine: MachineConfig = field(default_factory=MachineConfig)
    """Machine configuration"""
    axis: Literal['x', 'y', 'z', 'r', 'rho'] = 'x'
    """Which axis to evaluate gradient"""
    camera_idx: int = 0
    """Camera index in the eval dataset to render"""
    output_path: Path = Path("renders/output")
    """Path to output image file."""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    spp: int = 128
    """Render spp"""
    spp_per_batch: int = -1
    """Render spp per batch, to avoid MLE"""
    mi_config_name: str = 'warpone'
    """Method to be used for the optimization. Default: warpone"""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    target_value: float = 0.3
    """Gradient around the specified value."""
    detach_op: bool = False
    """Whether it detaches the nerf emitter op"""
    emitter_xml_path: Optional[Path] = None
    """Override emitter scene path"""
    camera_xml_path: Optional[Path] = None
    """Override camera path with mitsuba xml"""
    build_proposal: bool = False
    """Whether it invokes build_emitter_proposal"""
    sdf_cube_size: Optional[float] = None
    """Override sdf voxel with a cube"""
    guiding_type: Optional[Literal['vmf', 'env', 'emitter_xml']] = None
    """Path guiding type"""
    adjoint_sampling_strategy: Optional[Literal['primal']] = None
    """What emitter sampling strategy we use in adjoint rendering?"""
    query_emitter_index: Optional[int] = None
    """
    The emitter index to visualize, -1 means scene.environment()
    """
    bsdf_only: bool = False
    """
    Use bsdf sampling only
    """
    load_voxel_path: Optional[Path] = None
    """Load voxel at specific path"""
    sdf_res: Optional[int] = None
    """Mitsuba SDF resolution"""
    use_gradient_scaling: Optional[bool] = None
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    mock_rotation_id: Optional[int] = None
    """Mock the rotation id to another id"""
    apply_rotation_camera_file: Optional[bool] = False
    """Apply rotater rotations even if using a camera file"""
    mi_opt_config_name: Optional[str] = None
    """Optimization configurations to run"""

    def main(self) -> None:
        """Main function."""
        config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)

        os.makedirs(self.output_path.parent, exist_ok=True)

        if self.spp_per_batch < 0:
            self.spp_per_batch = self.spp
        config.machine = self.machine
        config.mixed_precision = self.mixed_precision
        config.pipeline.mi_config_name = self.mi_config_name
        config.pipeline.detach_op = self.detach_op
        config.forward_gradient = self
        if self.sdf_cube_size is not None:
            config.pipeline.sdf_cube_size = self.sdf_cube_size
        if self.emitter_xml_path is not None:
            config.pipeline.emitter_xml_path = self.emitter_xml_path
        if self.guiding_type is not None:
            config.pipeline.guiding_type = self.guiding_type
        if self.adjoint_sampling_strategy is not None:
            config.pipeline.adjoint_sampling_strategy = self.adjoint_sampling_strategy
        if self.query_emitter_index is not None:
            config.pipeline.query_emitter_index = self.query_emitter_index
        if self.sdf_res is not None:
            config.pipeline.sdf_res = self.sdf_res
        if self.use_gradient_scaling is not None:
            config.pipeline.model.use_gradient_scaling = self.use_gradient_scaling
        if self.apply_rotation_camera_file is not None:
            config.pipeline.apply_rotation_camera_file = self.apply_rotation_camera_file
        if self.mi_opt_config_name is not None:
            config.pipeline.mi_opt_config_name = self.mi_opt_config_name

        launch(
            main_func=main,
            num_devices_per_machine=config.machine.num_devices,
            device_type=config.machine.device_type,
            num_machines=config.machine.num_machines,
            machine_rank=config.machine.machine_rank,
            dist_url=config.machine.dist_url,
            config=config,
        )

    def render_forward_gradient(self, pipeline):
        if self.load_voxel_path is not None:
            if comms.is_main_process():
                pipeline.load_voxels(self.load_voxel_path, adaptive_resolution=False)
        if self.mock_rotation_id is not None:
            pipeline.rotater.rotation_ids[self.camera_idx] = self.mock_rotation_id
        if self.camera_xml_path is not None:
            cameras = mi.load_file(str(self.camera_xml_path)).sensors()
        else:
            cameras = pipeline.datamanager.eval_dataset.cameras
            cameras = cameras.to('cuda:0')
            cameras.rescale_output_resolution(1.0 / self.downscale_factor)

        if self.bsdf_only and comms.is_main_process():
            pipeline.sdf_scene.integrator()._use_adjoint = (False, False)
            pipeline.sdf_scene.integrator()._use_emitter = (False, False)
            pipeline.sdf_scene.integrator()._use_bsdf = (True, True)

        if self.build_proposal:
            pipeline.rescale_train_dataset(2511)
            pipeline.build_emitter_proposal()

        if comms.is_main_process():
            CONSOLE.print(f'Start rendering forward gradients at [bold][green]{self.output_path}[/bold]..')
        outputs = pipeline.render_forward_gradient(cameras, self.camera_idx, self.axis, self.spp, self.spp_per_batch,
                                                   self.target_value)

        if comms.is_main_process():
            table = Table(
                title=None,
                show_header=False,
                box=box.MINIMAL,
                title_style=style.Style(bold=True),
            )

            # Save outputs
            for name, output_image in outputs.items():
                this_output_path = f'{self.output_path}_{name}.exr'
                iio.imwrite(this_output_path, output_image.cpu().numpy())
                table.add_row("Image", this_output_path)

            CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ForwardGradient).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ForwardGradient)  # noqa
