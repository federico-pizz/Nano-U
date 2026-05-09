import os

from .metrics import BinaryIoU, NumpyEncoder
from .qat import NoOpQuantizeConfig, apply_qat_to_model

def get_project_root():
    # From src/utils/__init__.py, go up: utils -> src -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "BinaryIoU",
    "NumpyEncoder",
    "get_project_root",
    "NoOpQuantizeConfig",
    "apply_qat_to_model",
]
