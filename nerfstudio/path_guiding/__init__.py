from typing import Type

from nerfstudio.path_guiding.emitter_xml_guiding import EmitterXMLGuiding
from nerfstudio.path_guiding.env_guiding import EnvironmentGuiding
from nerfstudio.path_guiding.path_guiding import PathGuiding
from nerfstudio.path_guiding.vmf_guiding import VonMisesFisherGuiding

guiding_classes = {
    'vmf': VonMisesFisherGuiding,
    'env': EnvironmentGuiding,
    'emitter_xml': EmitterXMLGuiding,
}


def get_path_guiding_class(guiding_type) -> Type[PathGuiding]:
    return guiding_classes[guiding_type]
