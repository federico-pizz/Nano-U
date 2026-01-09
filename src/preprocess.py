"""Data preprocessing utilities.

Move or adapt the content of `data_preparation.py` into functions here
for a clean importable API used by training and inference scripts.
"""


def prepare_dataset(src_dir: str, dst_dir: str):
    print(f"Preparing dataset from {src_dir} to {dst_dir}")
