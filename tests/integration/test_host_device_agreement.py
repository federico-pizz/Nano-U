"""B4's acceptance check, host side: host and device scoring agree on the same model output.

The board cannot be in CI, so this feeds the INT8 model's own logits through the
device path - hex encoded the way firmware/src/bin/inference.rs prints them,
decoded by `read_device_outputs`, stacked by `device_probs` - and checks that
it scores exactly as the host path does. What it cannot check is the board's
own arithmetic against the TFLite interpreter's; that takes
`scripts/device_eval.py` on real hardware.
"""

import json
import os

import numpy as np
import pytest
import tf_keras as keras

from nanou import evaluate as ev
from nanou.config import resolve
from nanou.device.evaluate import device_probs, read_device_outputs
from nanou.predictor import KerasPredictor, TFLitePredictor, load_predictor, logits_to_probs
from nanou.quantize import convert_to_tflite_quantized

SHAPE = (60, 80, 3)
SCALARS = ("miou", "dice", "precision", "recall", "f0.5", "f1", "f2", "bce", "focal")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A small conv net with a linear head, saved as .h5 and exported as INT8. Seeded.

    Not Nano-U: fresh from init, Nano-U's output is around 1e-9 and INT8
    flattens it to a single value, which would make the agreement check
    vacuous. This net's logits spread over both signs. Nano-U's own export is
    covered in test_quantize.py; what is tested here is the scoring path.
    """
    keras.utils.set_random_seed(0)
    out = tmp_path_factory.mktemp("models")
    inp = keras.Input(SHAPE)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    model = keras.Model(inp, keras.layers.Conv2D(1, 1, name="output")(x))
    h5 = out / "nanou.h5"
    model.save(h5)

    rng = np.random.default_rng(0)

    def representative_gen():
        for _ in range(8):
            yield [rng.standard_normal((1, *SHAPE)).astype(np.float32)]

    tflite = out / "nanou.tflite"
    convert_to_tflite_quantized(model, str(tflite), representative_gen)
    return model, h5, tflite


@pytest.fixture
def preset(tmp_png_dataset, tmp_path):
    """A preset whose test split is the synthetic dataset; base.yaml gives SHAPE."""
    root = tmp_path / "data"
    (root / "test").mkdir(parents=True)
    (root / "test" / "img").symlink_to(os.path.dirname(tmp_png_dataset["imgs"][0]))
    (root / "test" / "mask").symlink_to(os.path.dirname(tmp_png_dataset["masks"][0]))
    path = tmp_path / "preset.yaml"
    path.write_text(
        f"extends: base\ndata_root: {root}\noutput_dir: {tmp_path / 'results'}\nnanou: {{}}\n"
    )
    return path


@pytest.fixture
def config(preset):
    return resolve(path=preset)


def test_the_suffix_picks_the_predictor(models):
    model, h5, tflite = models
    assert isinstance(load_predictor(tflite), TFLitePredictor)
    assert load_predictor(tflite).fmt == "int8"
    assert isinstance(load_predictor(h5), KerasPredictor)
    assert load_predictor(h5).fmt == "fp32"
    in_memory = load_predictor(model)
    assert isinstance(in_memory, KerasPredictor) and in_memory.path is None


def _constant_model(value):
    inp = keras.Input((2, 2, 1))
    return keras.Model(inp, keras.layers.Lambda(lambda t: t * 0.0 + value)(inp))


@pytest.mark.parametrize("value", [-3.0, 0.0, 0.4, 3.0])
def test_the_output_always_gets_the_sigmoid(value):
    # 0.0 and 0.4 are the regression cases: logits inside [0, 1] used to be
    # taken for probabilities on the host, while the device always applies
    # the sigmoid.
    x = np.zeros((1, 2, 2, 1), dtype=np.float32)
    expected = logits_to_probs(np.full((1, 2, 2, 1), value, dtype=np.float32))
    np.testing.assert_allclose(KerasPredictor(_constant_model(value)).predict(x), expected)


def test_host_and_device_paths_score_the_same_logits_identically(models, config):
    _, _, tflite = models
    imgs, masks = ev.split_files(config, "test")
    dataset = ev.split_dataset(config, imgs, masks)
    predictor = TFLitePredictor(tflite)
    host_probs, gt, _ = predictor.run(dataset)
    groups = ev.frame_groups(imgs, len(gt))
    host = ev.score(host_probs, gt, groups)

    # What the board prints for the same model: each image's dequantised
    # logits, one hex row at a time.
    logits = np.concatenate([predictor.raw(x.numpy()) for x, _ in dataset], axis=0)
    assert logits.min() < 0 < logits.max(), "degenerate fixture: the logits do not spread"
    lines = [f"BENCHMARK_START:{len(logits)}"]
    for i, image in enumerate(logits[..., 0]):
        lines.append(f"IMG_OUTPUT_START:{i}")
        lines += [row.astype("<f4").tobytes().hex() for row in image]
        lines.append(f"IMG_OUTPUT_END:{i}")
    lines.append("BENCHMARK_DONE")

    img_data = read_device_outputs(iter(lines), width=SHAPE[1])["img_data"]
    probs, kept = device_probs(img_data, len(gt), SHAPE[0])
    device = ev.score(probs, gt[kept], [groups[i] for i in kept])

    assert kept == list(range(len(gt)))
    np.testing.assert_array_equal(probs, host_probs)
    assert {k: device[k] for k in SCALARS} == {k: host[k] for k in SCALARS}


@pytest.mark.parametrize("which, fmt", [(1, "fp32"), (2, "int8")])
def test_evaluate_writes_two_figures_and_one_report(models, preset, tmp_path, which, fmt):
    out = tmp_path / "out"
    results = ev.evaluate(models[which], str(preset), out_dir=out)

    assert sorted(p.name for p in out.iterdir()) == [
        f"eval_{fmt}_test.json",
        f"eval_{fmt}_test.png",
        f"eval_{fmt}_test_metrics.png",
    ]
    report = json.loads((out / f"eval_{fmt}_test.json").read_text())
    assert (report["format"], report["split"]) == (fmt, "test")
    assert report["miou"] == results["miou"]
