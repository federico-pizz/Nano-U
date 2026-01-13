import os

from .data import make_dataset
from .metrics import BinaryIoU

def get_project_root():
    # From src/utils/__init__.py, go up: utils -> src -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["make_dataset", "BinaryIoU", "get_project_root"]
