"""The best-epoch checkpoint belongs to the run that wrote it.

A run with no validation split writes no checkpoint, so a file left in the run
directory by an earlier run must not be loaded as this run's best weights.
"""

import cv2
import numpy as np

from nanou.train import CHECKPOINT_FILE, train_model

H, W = 60, 80


def _write_split(root, split, n=4):
    rng = np.random.default_rng(0)
    for sub in ("img", "mask"):
        (root / split / sub).mkdir(parents=True)
    for i in range(n):
        name = f"seq_frame{i}.png"
        cv2.imwrite(
            str(root / split / "img" / name), (rng.random((H, W, 3)) * 255).astype(np.uint8)
        )
        cv2.imwrite(
            str(root / split / "mask" / name), ((rng.random((H, W)) > 0.5) * 255).astype(np.uint8)
        )


def _preset(tmp_path):
    data = tmp_path / "data"
    _write_split(data, "train")
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        f"extends: base\n"
        f"data_root: {data}\n"
        f"models_dir: {tmp_path / 'models'}\n"
        f"output_dir: {tmp_path / 'results'}\n"
        "epochs: 1\n"
        "batch_size: 2\n"
        "nanou:\n"
        "  filters: [4, 8, 16]\n"
        "  bottleneck: 16\n"
        "  decoder_filters: [16, 8, 4]\n"
        "  qat_enabled: false\n"
    )
    return preset


def test_a_stale_checkpoint_is_not_loaded(tmp_path):
    preset = _preset(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Not an HDF5 file: loading it would raise.
    (run_dir / CHECKPOINT_FILE).write_text("left behind by another run")

    result = train_model(str(preset), "nanou", output_dir=str(run_dir))

    assert (tmp_path / "models" / "nanou.h5").exists()
    assert not (run_dir / CHECKPOINT_FILE).exists()
    assert result["model_path"] == str(tmp_path / "models" / "nanou.h5")
