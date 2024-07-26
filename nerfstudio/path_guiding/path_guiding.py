from abc import abstractmethod
from typing import Dict, Any

import mitsuba as mi
import torch

from nerfstudio.pipelines.base_pipeline import Pipeline


def num_adjoint_emitters(adjoint_sampling_strategy):
    """Number of adjoint emitters, used for importance sampling of differential emitter distribution"""
    if adjoint_sampling_strategy == 'primal':
        return 0
    elif adjoint_sampling_strategy in ['adjoint', 'primal_adjoint']:
        return 1
    else:
        raise ValueError(f'unknown {adjoint_sampling_strategy}')


class PathGuiding:

    def __init__(self, sdf_scene, mi_config, adjoint_sampling_strategy, asset_path):
        self.sdf_scene = sdf_scene
        self.mi_config = mi_config
        self.adjoint_sampling_strategy = adjoint_sampling_strategy
        self.init_adjoint_emitters(adjoint_sampling_strategy, asset_path)

    @classmethod
    def emitter_scene_xml(cls, pipeline, asset_path):
        """
        """

    @classmethod
    def mts_args(cls, pipeline, asset_path):
        return {
            'emitter_scene': cls.emitter_scene_xml(pipeline, asset_path),
        }

    def init_adjoint_emitters(self, adjoint_sampling_strategy, asset_path):
        for i in range(num_adjoint_emitters(adjoint_sampling_strategy)):
            self.sdf_scene.integrator().adjoint_emitters.append(
                mi.load_file(f'{asset_path}/{self.emitter_scene_xml(None, asset_path)}').environment()
            )

    @abstractmethod
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        """

    @abstractmethod
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int):
        """
        """

    @abstractmethod
    def output_emitter_proposal(self, output_filename):
        """
        """

    @abstractmethod
    def build_emitter_proposal(self, pipeline: Pipeline):
        """
        """

    def train_primal(self, o: torch.Tensor, v: torch.Tensor, rgb: torch.Tensor, guiding_weight: torch.Tensor):
        pass

    def train_forward(self, o: torch.Tensor, v: torch.Tensor, grad_rgb: torch.Tensor, guiding_weight: torch.Tensor):
        pass

    def train_backward(self, o: torch.Tensor, v: torch.Tensor, o_grad: torch.Tensor, guiding_weight: torch.Tensor):
        pass

    def set_training_period(self, primal_training_period: int, adjoint_training_period: int):
        pass
