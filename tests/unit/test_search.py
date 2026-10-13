"""Unit tests for nanou.search - the leakage-safe CV search logic.

These cover the pure search/aggregation/selection helpers and a fully stubbed
``run_cv`` (``run_training`` monkeypatched), so no real training runs in CI.
"""

import inspect
from pathlib import Path

import pytest

from nanou import search as cv
from nanou.config import get_project_root

# -- make_config_overrides / CE->alpha mapping ----------------------------------


def test_ce_off_forces_alpha_one():
    cfg = cv.make_config_overrides(temperature=4.0, alpha=0.3, regime="none", ce_enabled=False)
    assert cfg["alpha"] == 1.0  # CE coefficient (1-alpha) becomes 0
    assert cfg["requested_alpha"] == 0.3
    assert cfg["ce_enabled"] is False


def test_ce_on_keeps_alpha():
    cfg = cv.make_config_overrides(temperature=4.0, alpha=0.3, regime="full", ce_enabled=True)
    assert cfg["alpha"] == 0.3
    assert cfg["ce_enabled"] is True


def test_regime_maps_to_augment_params():
    assert cv.make_config_overrides(2, 0.5, "none", True)["augment"] is False
    geo = cv.make_config_overrides(2, 0.5, "geometric", True)
    assert geo["augment"] is True and geo["augment_params"]["flip_prob"] == 0.5
    photo = cv.make_config_overrides(2, 0.5, "photometric", True)
    assert photo["augment_params"]["brightness"] > 0 and photo["augment_params"]["flip_prob"] == 0.0


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        cv.make_config_overrides(2, 0.5, "rotate-only", True)


# -- expand_grid (cartesian product + dedup) -----------------------------------


def test_expand_grid_size_ce_on_only():
    grid = cv.expand_grid([2, 4], [0.3, 0.7], ["none", "full"], [True])
    assert len(grid) == 2 * 2 * 2  # T x alpha x regime


def test_expand_grid_dedups_ce_off_alpha_collapse():
    # CE off forces alpha=1.0, so the two alpha values collapse to one per (T,regime).
    grid = cv.expand_grid([4], [0.3, 0.7], ["none"], [False])
    assert len(grid) == 1
    assert grid[0]["alpha"] == 1.0


def test_expand_grid_ce_on_and_off_distinct_rows():
    grid = cv.expand_grid([4], [0.5], ["none"], [True, False])
    keys = {(g["alpha"], g["ce_enabled"]) for g in grid}
    assert keys == {(0.5, True), (1.0, False)}


def test_expand_grid_tversky_axis():
    """tversky_weight multiplies the grid; loss-shaping keys only when w>0."""
    base = cv.expand_grid([4], [0.5], ["none"], [True])  # implicit [0.0]
    swept = cv.expand_grid([4], [0.5], ["none"], [True], [0.0, 0.5])
    assert len(swept) == 2 * len(base)  # extra axis
    by_w = {g["tversky_weight"]: g for g in swept}
    assert set(by_w) == {0.0, 0.5}
    assert "tversky_alpha" not in by_w[0.0]  # pure BCE, no keys
    assert by_w[0.5]["tversky_alpha"] == 0.7 and by_w[0.5]["tversky_beta"] == 0.3


def test_make_config_overrides_default_tversky_is_pure_bce():
    cfg = cv.make_config_overrides(4.0, 0.3, "none", True)
    assert cfg["tversky_weight"] == 0.0
    assert "tversky_alpha" not in cfg  # legacy config reproduced byte-for-byte


# -- aggregate_folds / select_best ---------------------------------------------


def test_aggregate_folds_mean_std():
    folds = [
        {
            "f0.5": 0.6,
            "f1": 0.5,
            "f2": 0.4,
            "precision": 0.6,
            "recall": 0.5,
            "miou": 0.7,
            "dice": 0.5,
        },
        {
            "f0.5": 0.8,
            "f1": 0.7,
            "f2": 0.6,
            "precision": 0.8,
            "recall": 0.7,
            "miou": 0.9,
            "dice": 0.7,
        },
    ]
    agg = cv.aggregate_folds(folds)
    assert agg["n_folds"] == 2
    assert abs(agg["f0.5_mean"] - 0.7) < 1e-9
    assert abs(agg["miou_mean"] - 0.8) < 1e-9
    assert abs(agg["f0.5_std"] - 0.1) < 1e-9


def test_select_best_default_is_miou_then_f05_tiebreak():
    """Default strategy: pick best mIoU; among equal mIoU prefer higher F0.5."""
    rows = [
        {"miou_mean": 0.90, "f0.5_mean": 0.50, "tag": "a"},
        {"miou_mean": 0.90, "f0.5_mean": 0.60, "tag": "b"},  # tie on miou, higher f0.5
        {"miou_mean": 0.88, "f0.5_mean": 0.99, "tag": "c"},
    ]
    assert cv.select_best(rows)["tag"] == "b"


def test_select_best_f05_primary_then_tiebreak():
    """Explicit F0.5-primary path (the overridable --select f0.5)."""
    rows = [
        {"f0.5_mean": 0.70, "miou_mean": 0.90, "tag": "a"},
        {"f0.5_mean": 0.70, "miou_mean": 0.95, "tag": "b"},  # tie on f0.5, higher miou
        {"f0.5_mean": 0.69, "miou_mean": 0.99, "tag": "c"},
    ]
    assert cv.select_best(rows, primary="f0.5_mean", secondary="miou_mean")["tag"] == "b"


def test_select_best_min_miou_floor_restricts_pool():
    rows = [
        {"f0.5_mean": 0.95, "miou_mean": 0.65, "tag": "high_f_low_miou"},  # below floor
        {"f0.5_mean": 0.80, "miou_mean": 0.72, "tag": "ok"},  # clears floor
        {"f0.5_mean": 0.78, "miou_mean": 0.90, "tag": "ok_lower_f"},
    ]
    # With F0.5-primary + floor: highest F0.5 is excluded; best eligible-by-F0.5 wins.
    assert (
        cv.select_best(rows, primary="f0.5_mean", secondary="miou_mean", min_miou=0.70)["tag"]
        == "ok"
    )


def test_select_best_raises_when_floor_infeasible():
    """The floor used to fall back to the full set, i.e. stop filtering."""
    rows = [
        {"f0.5_mean": 0.90, "miou_mean": 0.55, "tag": "a"},
        {"f0.5_mean": 0.60, "miou_mean": 0.68, "tag": "b"},
    ]
    with pytest.raises(ValueError, match=r"floor 0\.700 \(the best mean mIoU is 0\.6800\)"):
        cv.select_best(rows, primary="f0.5_mean", secondary="miou_mean", min_miou=0.70)


def test_a_grid_point_holds_schema_keys_and_row_only_keys_nothing_else():
    """What reaches train_model must resolve; a new record-only key has to be declared."""
    from nanou.config import TRAIN_KEYS

    point = cv.make_config_overrides(0.7, 0.3, "photometric", False, 0.7, 0.8, 0.2)
    assert set(cv.ROW_ONLY_KEYS) <= set(point)
    assert set(point) - set(cv.ROW_ONLY_KEYS) <= TRAIN_KEYS


# -- run_cv end-to-end with stubbed training (no TF training) -------------------


def test_run_cv_stubbed(monkeypatch, tmp_path):
    """run_cv assembles a table + winner without any real training."""
    files = [f"/d/img/s{s}_frame{f}.png" for s in range(4) for f in range(3)]
    masks = [p.replace("/img/", "/mask/") for p in files]
    monkeypatch.setattr(cv, "pool_train_val", lambda _c: (files, masks))

    # Stub training: just return a fake saved-model path per call.
    calls = {"n": 0}

    def fake_run_training(name, cfg_path, out_dir, config_overrides=None):
        calls["n"] += 1
        return {"model_path": f"{out_dir}/{name}.h5"}

    monkeypatch.setattr(cv, "run_training", fake_run_training)

    # Stub eval: f0.5/miou depend on temperature so a clear winner emerges.
    def fake_eval(model_path, img_files, mask_files, config, threshold=0.5):
        # config_overrides aren't passed here; derive a deterministic score from
        # the run directory name which encodes the tag (temperature).
        score = 0.9 if "T8" in model_path else 0.5
        return {
            "f0.5": score,
            "f1": score,
            "f2": score,
            "precision": score,
            "recall": score,
            "miou": score,
            "dice": score,
        }

    monkeypatch.setattr(cv, "evaluate_on_files", fake_eval)

    grid = cv.expand_grid([2.0, 8.0], [0.5], ["none"], [True])
    res = cv.run_cv(
        "config/presets/tinyagri.yaml",
        k=2,
        epochs=1,
        grid=grid,
        output_dir=str(tmp_path),
        fixed_teacher="/fake/teacher.h5",
    )

    assert len(res["rows"]) == 2
    assert res["best"]["temperature"] == 8.0  # higher-scoring config wins
    assert (tmp_path / "cv_results.json").exists()
    assert (tmp_path / "cv_results.csv").exists()
    # 2 configs x 2 folds x 1 student run each (teacher fixed) = 4 training calls
    assert calls["n"] == 4


def test_run_cv_caches_one_teacher_per_fold(monkeypatch, tmp_path):
    """Without a fixed teacher: teachers trained == k, not k x n_configs.

    Locks the caching optimization - the per-fold teacher depends only on the
    leakage-safe split, so it is trained once and reused across every config.
    """
    files = [f"/d/img/s{s}_frame{f}.png" for s in range(4) for f in range(3)]
    masks = [p.replace("/img/", "/mask/") for p in files]
    monkeypatch.setattr(cv, "pool_train_val", lambda _c: (files, masks))

    counts = {"bu_net": 0, "nanou": 0}

    def fake_run_training(name, cfg_path, out_dir, config_overrides=None):
        counts[name] += 1
        # Teacher path must be per-fold-unique so students reuse the right one.
        return {"model_path": f"{out_dir}/{name}.h5"}

    monkeypatch.setattr(cv, "run_training", fake_run_training)
    monkeypatch.setattr(
        cv,
        "evaluate_on_files",
        lambda *a, **k: {
            m: 0.5 for m in ("f0.5", "f1", "f2", "precision", "recall", "miou", "dice")
        },
    )

    grid = cv.expand_grid([2.0, 8.0], [0.5], ["none"], [True])  # 2 configs
    cv.run_cv(
        "config/presets/tinyagri.yaml", k=2, epochs=1, grid=grid, output_dir=str(tmp_path)
    )  # no fixed teacher

    assert counts["bu_net"] == 2  # k folds - NOT 2 configs x 2 folds == 4
    assert counts["nanou"] == 4  # 2 configs x 2 folds


# -- resume ---------------------------------------------------------------------


def _stub_training(monkeypatch, kill_student_at=None):
    """Stub training that saves a model file, as the real one does, and counts runs.

    With *kill_student_at* the n-th student run dies before it finishes, which
    is what a walltime does to a sweep.
    """
    files = [f"/d/img/s{s}_frame{f}.png" for s in range(4) for f in range(3)]
    masks = [p.replace("/img/", "/mask/") for p in files]
    monkeypatch.setattr(cv, "pool_train_val", lambda _c: (files, masks))
    counts = {"bu_net": 0, "nanou": 0}

    def fake_run_training(name, cfg_path, out_dir, config_overrides=None):
        counts[name] += 1
        if name == "nanou" and counts[name] == kill_student_at:
            raise RuntimeError("killed")
        path = Path(out_dir) / f"{name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return {"model_path": str(path)}

    def fake_eval(model_path, img_files, mask_files, config, threshold=0.5):
        score = 0.9 if "T8" in model_path else 0.5
        return {m: score for m in ("f0.5", "f1", "f2", "precision", "recall", "miou", "dice")}

    monkeypatch.setattr(cv, "run_training", fake_run_training)
    monkeypatch.setattr(cv, "evaluate_on_files", fake_eval)
    return counts


def _cv(tmp_path, epochs=1):
    grid = cv.expand_grid([2.0, 8.0], [0.5], ["none"], [True])
    return cv.run_cv(
        "config/presets/tinyagri.yaml", k=2, epochs=epochs, grid=grid, output_dir=str(tmp_path)
    )


def test_a_rerun_reuses_every_finished_run(monkeypatch, tmp_path):
    _stub_training(monkeypatch)
    first = _cv(tmp_path)
    counts = _stub_training(monkeypatch)
    second = _cv(tmp_path)
    assert counts == {"bu_net": 0, "nanou": 0}
    assert second["rows"] == first["rows"]


def test_a_killed_sweep_resumes_with_what_it_had_not_finished(monkeypatch, tmp_path):
    _stub_training(monkeypatch, kill_student_at=3)
    with pytest.raises(RuntimeError, match="killed"):
        _cv(tmp_path)
    counts = _stub_training(monkeypatch)
    res = _cv(tmp_path)
    # Both teachers and the first two students finished before the kill.
    assert counts == {"bu_net": 0, "nanou": 2}
    assert res["best"]["temperature"] == 8.0


def test_a_run_whose_job_changed_is_trained_again(monkeypatch, tmp_path):
    _stub_training(monkeypatch)
    _cv(tmp_path, epochs=1)
    counts = _stub_training(monkeypatch)
    _cv(tmp_path, epochs=2)
    assert counts == {"bu_net": 2, "nanou": 4}


def test_a_teacher_whose_model_is_gone_is_trained_again(monkeypatch, tmp_path):
    _stub_training(monkeypatch)
    _cv(tmp_path)
    (tmp_path / "teachers" / "fold0" / "bu_net.h5").unlink()
    counts = _stub_training(monkeypatch)
    _cv(tmp_path)
    assert counts["bu_net"] == 1


# -- load_sweep (sweeps/*.yaml) --------------------------------------------------

SWEEPS = get_project_root() / "sweeps"

# Configs per shipped sweep file, counted in the results each file was rebuilt
# from (the saved cv_results.json, or the run directories where the table was
# overwritten) - so the files describe the sweeps that actually ran.
SHIPPED_SWEEPS = {
    "tinyagri-stage1": 8,
    "tinyagri-stage2": 9,
    "tinyagri-stage3": 4,
    "tinyagri-stage3b": 3,
    "botanicgarden-stage2": 18,
    "botanicgarden-stage3": 8,
}

# What scripts/sweep.py adds itself; everything else comes from the file.
MACHINE_ARGS = {"config_path", "output_dir", "jobs", "teacher_jobs"}

MINIMAL = """\
k: 2
epochs: 1
grid:
  temperature: [4.0]
  alpha: [0.3]
  regime: [none]
  ce: [on]
"""


def _sweep_file(tmp_path, text):
    path = tmp_path / "sweep.yaml"
    path.write_text(text)
    return path


def test_every_shipped_sweep_has_a_known_size():
    assert sorted(p.stem for p in SWEEPS.glob("*.yaml")) == sorted(SHIPPED_SWEEPS)


@pytest.mark.parametrize("name, n_configs", sorted(SHIPPED_SWEEPS.items()))
def test_shipped_sweeps_expand_to_the_grids_that_ran(name, n_configs):
    sweep = cv.load_sweep(SWEEPS / f"{name}.yaml")
    assert len(sweep["grid"]) == n_configs
    assert set(sweep) == set(inspect.signature(cv.run_cv).parameters) - MACHINE_ARGS


def test_a_minimal_sweep_takes_the_defaults(tmp_path):
    sweep = cv.load_sweep(_sweep_file(tmp_path, MINIMAL))
    assert (sweep["k"], sweep["epochs"], sweep["seed"]) == (2, 1, 37)
    assert (sweep["select_primary"], sweep["min_miou"]) == ("miou_mean", 0.0)
    assert sweep["fixed_teacher"] is None
    # Leaving the tversky axis out means plain BCE.
    assert [g["tversky_weight"] for g in sweep["grid"]] == [0.0]


def test_ce_takes_on_and_off_quoted_or_not(tmp_path):
    text = MINIMAL.replace("ce: [on]", "ce: [on, 'off']")
    grid = cv.load_sweep(_sweep_file(tmp_path, text))["grid"]
    assert [g["ce_enabled"] for g in grid] == [True, False]


def test_tversky_takes_triples_and_bare_weights(tmp_path):
    text = MINIMAL + "  tversky:\n    - 0.5\n    - [0.7, 0.8, 0.2]\n"
    grid = cv.load_sweep(_sweep_file(tmp_path, text))["grid"]
    specs = [(g["tversky_weight"], g["tversky_alpha"], g["tversky_beta"]) for g in grid]
    assert specs == [(0.5, 0.7, 0.3), (0.7, 0.8, 0.2)]


@pytest.mark.parametrize(
    "text, match",
    [
        (MINIMAL + "epoch: 5\n", "unknown key"),
        (MINIMAL.replace("k: 2\n", ""), "missing required"),
        (MINIMAL + "  temprature: [2.0]\n", "unknown grid axis"),
        (MINIMAL.replace("regime: [none]", "regime: []"), "non-empty"),
        (MINIMAL.replace("ce: [on]", "ce: [maybe]"), "ce takes on/off"),
        (MINIMAL + "select: iou\n", "select must be"),
        (MINIMAL + "  tversky: [[0.7, 0.8]]\n", "triple"),
    ],
)
def test_a_malformed_sweep_raises(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        cv.load_sweep(_sweep_file(tmp_path, text))
