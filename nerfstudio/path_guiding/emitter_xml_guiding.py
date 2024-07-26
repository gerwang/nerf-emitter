import os
from typing import Dict, Any

from nerfstudio.pipelines.base_pipeline import Pipeline
from .path_guiding import PathGuiding


class EmitterXMLGuiding(PathGuiding):

    @classmethod
    def emitter_scene_xml(cls, pipeline, asset_path):
        return os.path.relpath(pipeline.config.emitter_xml_path, asset_path)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        pass

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int):
        pass

    def output_emitter_proposal(self, output_filename):
        pass

    def build_emitter_proposal(self, pipeline: Pipeline):
        pass
