import os
from typing import Dict, Any

from nerfstudio.pipelines.base_pipeline import Pipeline
from .path_guiding import PathGuiding


class EnvironmentGuiding(PathGuiding):

    @classmethod
    def emitter_scene_xml(cls, pipeline, asset_path):
        return 'emitters/dataset_env.xml'

    @classmethod
    def mts_args(cls, pipeline, asset_path):
        res = super().mts_args(pipeline, asset_path)
        res.update({
            'envmap_filename': os.path.relpath(pipeline.datamanager.config.dataparser.data / 'env.exr', asset_path),
        })
        return res

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        pass

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int):
        pass

    def output_emitter_proposal(self, output_filename):
        pass

    def build_emitter_proposal(self, pipeline: Pipeline):
        pass
