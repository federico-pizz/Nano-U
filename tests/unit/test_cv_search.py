"""Unit tests for scripts/cv_search.py — the leakage-safe CV search logic.

These cover the pure search/aggregation/selection helpers and a fully stubbed
``run_cv`` (``run_training`` monkeypatched), so no real training runs in CI.
"""

import importlib

import numpy as np
import pytest

cv = importlib.import_module("scripts.cv_search")


# ── make_config_overrides / CE→alpha mapping (#7) ─────────────────────────────

def test_ce_off_forces_alpha_one():
    cfg = cv.make_config_overrides(temperature=4.0, alpha=0.3, regime="none", ce_enabled=False)
    assert cfg["alpha"] == 1.0          # CE coefficient (1-alpha) becomes 0
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


# ── expand_grid (cartesian product + dedup) ───────────────────────────────────

def test_expand_grid_size_ce_on_only():
    grid = cv.expand_grid([2, 4], [0.3, 0.7], ["none", "full"], [True])
    assert len(grid) == 2 * 2 * 2  # T × alpha × regime


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
    base = cv.expand_grid([4], [0.5], ["none"], [True])              # implicit [0.0]
    swept = cv.expand_grid([4], [0.5], ["none"], [True], [0.0, 0.5])
    assert len(swept) == 2 * len(base)                              # extra axis
    by_w = {g["tversky_weight"]: g for g in swept}
    assert set(by_w) == {0.0, 0.5}
    assert "tversky_alpha" not in by_w[0.0]                         # pure BCE, no keys
    assert by_w[0.5]["tversky_alpha"] == 0.7 and by_w[0.5]["tversky_beta"] == 0.3


def test_make_config_overrides_default_tversky_is_pure_bce():
    cfg = cv.make_config_overrides(4.0, 0.3, "none", True)
    assert cfg["tversky_weight"] == 0.0
    assert "tversky_alpha" not in cfg  # legacy config reproduced byte-for-byte


# ── aggregate_folds / select_best ─────────────────────────────────────────────

def test_aggregate_folds_mean_std():
    folds = [
        {"f0.5": 0.6, "f1": 0.5, "f2": 0.4, "precision": 0.6, "recall": 0.5, "miou": 0.7, "dice": 0.5},
        {"f0.5": 0.8, "f1": 0.7, "f2": 0.6, "precision": 0.8, "recall": 0.7, "miou": 0.9, "dice": 0.7},
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
        {"f0.5_mean": 0.80, "miou_mean": 0.72, "tag": "ok"},               # clears floor
        {"f0.5_mean": 0.78, "miou_mean": 0.90, "tag": "ok_lower_f"},
    ]
    # With F0.5-primary + floor: highest F0.5 is excluded; best eligible-by-F0.5 wins.
    assert cv.select_best(rows, primary="f0.5_mean", secondary="miou_mean",
                          min_miou=0.70)["tag"] == "ok"


def test_select_best_falls_back_when_floor_infeasible():
    rows = [
        {"f0.5_mean": 0.90, "miou_mean": 0.55, "tag": "a"},
        {"f0.5_mean": 0.60, "miou_mean": 0.68, "tag": "b"},
    ]
    # No config clears 0.70 → fall back to full set; best by F0.5 (tiebreak miou).
    assert cv.select_best(rows, primary="f0.5_mean", secondary="miou_mean",
                          min_miou=0.70)["tag"] == "a"


# ── run_cv end-to-end with stubbed training (no TF training) ───────────────────

def test_run_cv_stubbed(monkeypatch, tmp_path):
    """run_cv assembles a table + winner without any real training."""
    files = [f"/d/img/s{s}_frame{f}.png" for s in range(4) for f in range(3)]
    masks = [p.replace("/img/", "/mask/") for p in files]

    monkeypatch.setattr(cv, "load_config", lambda _p: {
        "data": {"normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
                 "input_shape": [60, 80, 3]}})
    monkeypatch.setattr(cv, "pool_train_val", lambda _c: (files, masks))

    # Stub training: just return a fake saved-model path per call.
    calls = {"n": 0}

    def fake_run_training(name, cfg_path, out_dir, config_overrides=None):
        calls["n"] += 1
        return {"status": "success", "model_path": f"{out_dir}/{name}.h5"}

    monkeypatch.setattr(cv, "run_training", fake_run_training)

    # Stub eval: f0.5/miou depend on temperature so a clear winner emerges.
    def fake_eval(model_path, img_files, mask_files, config, threshold=0.5):
        # config_overrides aren't passed here; derive a deterministic score from
        # the run directory name which encodes the tag (temperature).
        score = 0.9 if "T8" in model_path else 0.5
        return {"f0.5": score, "f1": score, "f2": score,
                "precision": score, "recall": score, "miou": score, "dice": score}

    monkeypatch.setattr(cv, "evaluate_on_files", fake_eval)

    grid = cv.expand_grid([2.0, 8.0], [0.5], ["none"], [True])
    res = cv.run_cv("config/config.yaml", k=2, epochs=1, grid=grid,
                    output_dir=str(tmp_path), fixed_teacher="/fake/teacher.h5")

    assert len(res["rows"]) == 2
    assert res["best"]["temperature"] == 8.0          # higher-scoring config wins
    assert (tmp_path / "cv_results.json").exists()
    assert (tmp_path / "cv_results.csv").exists()
    # 2 configs × 2 folds × 1 student run each (teacher fixed) = 4 training calls
    assert calls["n"] == 4


def test_run_cv_caches_one_teacher_per_fold(monkeypatch, tmp_path):
    """Without a fixed teacher: teachers trained == k, not k × n_configs.

    Locks the caching optimization — the per-fold teacher depends only on the
    leakage-safe split, so it is trained once and reused across every config.
    """
    files = [f"/d/img/s{s}_frame{f}.png" for s in range(4) for f in range(3)]
    masks = [p.replace("/img/", "/mask/") for p in files]
    monkeypatch.setattr(cv, "load_config", lambda _p: {
        "data": {"normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
                 "input_shape": [60, 80, 3]}})
    monkeypatch.setattr(cv, "pool_train_val", lambda _c: (files, masks))

    counts = {"bu_net": 0, "nano_u": 0}

    def fake_run_training(name, cfg_path, out_dir, config_overrides=None):
        counts[name] += 1
        # Teacher path must be per-fold-unique so students reuse the right one.
        return {"status": "success", "model_path": f"{out_dir}/{name}.h5"}

    monkeypatch.setattr(cv, "run_training", fake_run_training)
    monkeypatch.setattr(cv, "evaluate_on_files",
                        lambda *a, **k: {m: 0.5 for m in
                                         ("f0.5", "f1", "f2", "precision",
                                          "recall", "miou", "dice")})

    grid = cv.expand_grid([2.0, 8.0], [0.5], ["none"], [True])  # 2 configs
    cv.run_cv("config/config.yaml", k=2, epochs=1, grid=grid,
              output_dir=str(tmp_path))                          # no fixed teacher

    assert counts["bu_net"] == 2          # k folds — NOT 2 configs × 2 folds == 4
    assert counts["nano_u"] == 4          # 2 configs × 2 folds
