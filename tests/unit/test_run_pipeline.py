"""Failures raise, and both pipeline paths write one summary schema.

Training, export and evaluation are stubbed, so these check the orchestration
only: what is called, with what, and what lands on disk.
"""

import json
from pathlib import Path

import pytest

from nanou import pipeline

PRESET = "config/presets/tinyagri.yaml"

ENTRY_KEYS = {
    "model_path",
    "run_dir",
    "qat_applied",
    "tflite_path",
    "tflite_size_kb",
    "quant_params_path",
    "eval_fp32",
    "eval_int8",
}


@pytest.fixture
def stubs(monkeypatch):
    """Replace the three expensive steps; record what each was asked to do."""
    import nanou.evaluate

    calls = {"train": [], "export": [], "eval": []}

    def fake_run_training(name, config_path, output_dir, config_overrides=None):
        calls["train"].append((name, config_overrides))
        return {
            "model_name": name,
            "model_path": f"{output_dir}/{name}.h5",
            "experiment_dir": f"{output_dir}/{name}",
            "qat_applied": name != pipeline.TEACHER,
        }

    def fake_export(keras_path, models_dir, config_path):
        calls["export"].append(keras_path.stem)
        return {
            "tflite_path": f"{models_dir}/{keras_path.stem}.tflite",
            "quant_params_path": f"{models_dir}/{keras_path.stem}_quant_params.json",
            "size_kb": 12.5,
        }

    def fake_evaluate(model_path, config_path, out_dir):
        fmt = "int8" if str(model_path).endswith(".tflite") else "fp32"
        calls["eval"].append((Path(model_path).stem, fmt))
        return {"miou": 0.8 if fmt == "int8" else 0.9}

    monkeypatch.setattr(pipeline, "run_training", fake_run_training)
    monkeypatch.setattr(pipeline, "export_int8", fake_export)
    monkeypatch.setattr(nanou.evaluate, "evaluate", fake_evaluate)
    return calls


def test_a_broken_config_raises_instead_of_returning_a_status(tmp_path):
    """The acceptance check: no `{"status": "failed"}`, the real exception instead."""
    with pytest.raises(KeyError, match="no_such_experiment"):
        pipeline.run_training("no_such_experiment", PRESET, str(tmp_path))


@pytest.mark.parametrize("overrides, seed", [(None, 98), ({"seed": 7}, 7)])
def test_the_run_seed_reaches_the_rngs(monkeypatch, tmp_path, overrides, seed):
    """seed_variance.py depends on this: an override used to lose to the file's seed."""
    import nanou.train

    class Seeded(Exception):
        pass

    def record(value):
        raise Seeded(value)

    monkeypatch.setattr(nanou.train, "set_global_seed", record)
    with pytest.raises(Seeded) as caught:
        pipeline.run_training(pipeline.STUDENT, PRESET, str(tmp_path), overrides)
    assert caught.value.args == (seed,)


def test_quantize_raises_on_a_missing_model(tmp_path):
    from nanou.quantize import quantize_model

    with pytest.raises(FileNotFoundError, match="No model to quantize"):
        quantize_model(str(tmp_path / "absent.h5"), str(tmp_path / "absent.tflite"), PRESET)


def test_qad_distils_from_the_teacher_it_just_trained(stubs, tmp_path):
    pipeline.run_pipeline(PRESET, str(tmp_path))
    (teacher, _), (student, overrides) = stubs["train"]
    assert (teacher, student) == (pipeline.TEACHER, pipeline.STUDENT)
    assert overrides == {"teacher_weights": f"{tmp_path}/{pipeline.TEACHER}.h5"}


def test_qad_exports_and_int8_scores_the_student_only(stubs, tmp_path):
    summary = pipeline.run_pipeline(PRESET, str(tmp_path))
    assert stubs["export"] == [pipeline.STUDENT]
    assert sorted(stubs["eval"]) == [
        (pipeline.TEACHER, "fp32"),
        (pipeline.STUDENT, "fp32"),
        (pipeline.STUDENT, "int8"),
    ]
    teacher = summary["models"][pipeline.TEACHER]
    assert teacher["tflite_path"] is None and teacher["eval_int8"] is None
    assert summary["models"][pipeline.STUDENT]["eval_int8"] == {"miou": 0.8}


def test_both_paths_write_the_same_schema_to_the_same_place(stubs, tmp_path):
    qad = pipeline.run_pipeline(PRESET, str(tmp_path / "qad"))
    single = pipeline.run_pipeline(
        PRESET, str(tmp_path / "single"), distil=False, model=pipeline.STUDENT
    )

    for summary, distil in ((qad, True), (single, False)):
        assert set(summary) == {"config_path", "distil", "models"}
        assert summary["distil"] is distil
        for entry in summary["models"].values():
            assert set(entry) == ENTRY_KEYS

    assert set(qad["models"]) == {pipeline.TEACHER, pipeline.STUDENT}
    assert set(single["models"]) == {pipeline.STUDENT}
    for sub, summary in (("qad", qad), ("single", single)):
        on_disk = json.loads((tmp_path / sub / pipeline.SUMMARY_FILE).read_text())
        assert on_disk == summary


def test_no_distill_turns_distillation_off(stubs, tmp_path):
    pipeline.run_pipeline(PRESET, str(tmp_path), distil=False, model=pipeline.TEACHER)
    assert stubs["train"] == [(pipeline.TEACHER, {"use_distillation": False})]
    assert stubs["export"] == []


@pytest.mark.parametrize("distil, model", [(True, "nanou"), (False, None)])
def test_distil_and_model_must_agree(distil, model, tmp_path):
    with pytest.raises(ValueError):
        pipeline.run_pipeline(PRESET, str(tmp_path), distil=distil, model=model)


def test_a_failed_export_propagates_and_writes_no_summary(stubs, monkeypatch, tmp_path):
    def broken_export(*_a, **_k):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(pipeline, "export_int8", broken_export)
    with pytest.raises(RuntimeError, match="converter exploded"):
        pipeline.run_pipeline(PRESET, str(tmp_path))
    assert not (tmp_path / pipeline.SUMMARY_FILE).exists()
