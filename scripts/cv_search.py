"""Leakage-safe cross-validation hyperparameter search for Nano-U.

Selects distillation/augmentation hyperparameters *honestly*:

  * Folds are built by whole capture sequence (``sequence_group``) with
    :func:`src.data.grouped_kfold`, so no scene is split across the train and
    validation side of a fold — adjacent near-duplicate frames cannot leak.
  * The search runs over the **pooled train+val** frames only. The test split is
    never touched here; it is measured once, afterwards, by ``evaluate_and_plot``.
  * Each (fold, config) trains through the normal ``run_training`` code path
    (QAT + distillation), with a **per-fold teacher** so the teacher never sees
    its fold's validation frames either.

Primary selection criterion is **mIoU** — a neutral, balanced overlap metric, so
the config choice is not circular with Nano-U's conservative (precision-favoring)
objective: conservatism is designed into the *loss*, not manufactured by the
selector. The precision-weighted **F0.5** is the tiebreaker and is reported as the
safety axis; all of F0.5/F1/F2/precision/recall/mIoU/Dice are logged per fold and
aggregated (mean ± std) into ``results/<dataset>/cv_results.{csv,json}``.

This is built to run unattended for hours. Example:

    python scripts/cv_search.py --config config/config.yaml --k 5 --epochs 50 \
        --temperatures 2 4 8 --alphas 0.3 0.5 0.7 \
        --regimes none geometric photometric full --ce on off
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data import make_dataset, sorted_by_frame, sequence_group, grouped_kfold
from src.pipeline import run_training
from src.utils.config import load_config
from src.utils import get_project_root
from src.evaluate import run_inference, compute_segmentation_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pure, unit-testable search logic (no training, no TF state)
# ─────────────────────────────────────────────────────────────────────────────

# Named augmentation regimes → make_dataset kwargs (the #8 ablation axis).
AUGMENT_REGIMES: Dict[str, Dict[str, Any]] = {
    "none": {"augment": False, "augment_params": {}},
    "geometric": {"augment": True, "augment_params": {
        "flip_prob": 0.5, "max_rotation_deg": 20.0,
        "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0}},
    "photometric": {"augment": True, "augment_params": {
        "flip_prob": 0.0, "max_rotation_deg": 0.0,
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}},
    "full": {"augment": True, "augment_params": {
        "flip_prob": 0.5, "max_rotation_deg": 20.0,
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}},
}


def make_config_overrides(temperature: float, alpha: float, regime: str,
                          ce_enabled: bool, tversky_weight: float = 0.0,
                          tversky_alpha: float = 0.7,
                          tversky_beta: float = 0.3) -> Dict[str, Any]:
    """Map one grid point to the config_overrides train_model understands.

    CE on/off (#7) is expressed through the existing KD weighting: with the loss
    ``(1-alpha)·CE + alpha·distill``, disabling CE is exactly ``alpha = 1.0``.
    The requested alpha is kept (as ``requested_alpha``) for traceability.

    ``tversky_weight`` (default 0.0 = pure BCE) blends the precision-favoring
    Tversky term into the student's supervised loss
    (``(1-w)·BCE + w·Tversky(alpha_fp, beta_fn)``). It is always recorded on the
    row; the loss-shaping keys are only injected when the term is active so a
    weight of 0 reproduces the legacy BCE config byte-for-byte.
    """
    if regime not in AUGMENT_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; "
                         f"choose from {sorted(AUGMENT_REGIMES)}")
    reg = AUGMENT_REGIMES[regime]
    effective_alpha = alpha if ce_enabled else 1.0
    cfg = {
        "temperature": float(temperature),
        "alpha": float(effective_alpha),
        "ce_enabled": bool(ce_enabled),
        "requested_alpha": float(alpha),
        "augment_regime": regime,
        "augment": reg["augment"],
        "augment_params": dict(reg["augment_params"]),
        "tversky_weight": float(tversky_weight),
    }
    if tversky_weight > 0.0:
        cfg["tversky_alpha"] = float(tversky_alpha)
        cfg["tversky_beta"] = float(tversky_beta)
    return cfg


def expand_grid(temperatures: List[float], alphas: List[float],
                regimes: List[str], ce_options: List[bool],
                tversky_weights: List[float] = (0.0,)) -> List[Dict[str, Any]]:
    """Cartesian product → config_overrides, de-duplicated.

    When CE is off, alpha is forced to 1.0, so the alpha axis collapses; identical
    effective configs (same temperature/effective-alpha/regime/ce/tversky) are
    emitted once. ``tversky_weights`` defaults to ``(0.0,)`` so callers that don't
    sweep the conservative loss get exactly the previous grid.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for T, a, reg, ce, tw in itertools.product(
            temperatures, alphas, regimes, ce_options, tversky_weights):
        cfg = make_config_overrides(T, a, reg, ce, tversky_weight=tw)
        key = (cfg["temperature"], cfg["alpha"], cfg["augment_regime"],
               cfg["ce_enabled"], cfg["tversky_weight"])
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def aggregate_folds(fold_metrics: List[Dict[str, float]],
                    keys=("f0.5", "f1", "f2", "precision", "recall", "miou", "dice"),
                    ) -> Dict[str, float]:
    """Mean ± std across folds for each metric → flat ``<key>_mean/_std`` dict."""
    agg: Dict[str, float] = {"n_folds": len(fold_metrics)}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=np.float64)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    return agg


def build_row(cfg: Dict[str, Any], agg: Dict[str, float],
              fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """One results table row: the config dims + aggregates + per-fold headline."""
    row = {
        "temperature": cfg["temperature"],
        "alpha": cfg["alpha"],
        "requested_alpha": cfg["requested_alpha"],
        "ce_enabled": cfg["ce_enabled"],
        "augment_regime": cfg["augment_regime"],
        "tversky_weight": cfg.get("tversky_weight", 0.0),
    }
    row.update(agg)
    row["fold_f0.5"] = [round(fm["f0.5"], 5) for fm in fold_metrics]
    row["fold_miou"] = [round(fm["miou"], 5) for fm in fold_metrics]
    return row


def select_best(rows: List[Dict[str, Any]], primary: str = "miou_mean",
                secondary: str = "f0.5_mean", min_miou: float = 0.0) -> Dict[str, Any]:
    """Pick the row maximizing the primary metric, breaking ties on the secondary.

    Default is mIoU-primary with F0.5 as the tiebreak: keep the best-overlap
    config, and among equal-overlap configs prefer the more precision-favoring
    one. With ``min_miou > 0`` the choice is restricted to configs whose mean mIoU
    clears the floor (useful only when a non-mIoU ``primary`` is requested). If no
    config clears the floor the selection falls back to the full set; the caller
    is expected to log that the floor was infeasible.
    """
    if not rows:
        raise ValueError("no rows to select from")
    eligible = [r for r in rows if r.get("miou_mean", 0.0) >= min_miou]
    pool = eligible or rows
    return max(pool, key=lambda r: (r[primary], r[secondary]))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration (real training; exercised manually, stubbed in tests)
# ─────────────────────────────────────────────────────────────────────────────

def _list_split_files(config: Dict[str, Any], split: str):
    """Resolve (img_files, mask_files) for a processed split, frame-sorted."""
    root = str(get_project_root())

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    sc = config["data"]["paths"]["processed"][split]
    img_dir, mask_dir = rp(sc["img"]), rp(sc["mask"])
    imgs = sorted_by_frame([os.path.join(img_dir, f)
                            for f in os.listdir(img_dir) if f.endswith(".png")])
    masks = sorted_by_frame([os.path.join(mask_dir, f)
                             for f in os.listdir(mask_dir) if f.endswith(".png")])
    return imgs, masks


def pool_train_val(config: Dict[str, Any]):
    """Pool the train and val processed splits (test stays untouched)."""
    ti, tm = _list_split_files(config, "train")
    try:
        vi, vm = _list_split_files(config, "val")
    except (KeyError, FileNotFoundError):
        vi, vm = [], []
    return ti + vi, tm + vm


def evaluate_on_files(model_path: str, img_files, mask_files,
                      config: Dict[str, Any], threshold: float = 0.5) -> Dict[str, float]:
    """Build a dataset from explicit files and score a saved model on it."""
    norm = config["data"]["normalization"]
    ishape = config["data"].get("input_shape", [60, 80, 3])
    ds = make_dataset(img_files, mask_files, batch_size=8, shuffle=False,
                      augment=False, target_size=(ishape[0], ishape[1]),
                      mean=norm["mean"], std=norm["std"])
    probs, masks, _ = run_inference(model_path, ds)
    groups = [sequence_group(p) for p in sorted_by_frame(img_files)]
    if len(groups) != probs.shape[0]:
        groups = None
    return compute_segmentation_metrics(probs, masks, groups=groups,
                                        operating_threshold=threshold)


# Module-level worker jobs so they are picklable by the 'spawn' process pool.
# Each writes to its own models_dir (passed in config_overrides) so concurrent
# jobs never clobber a shared temp_model.h5 / <model>.h5.

def _train_teacher_job(job: Dict[str, Any]):
    """Train one per-fold BU-Net teacher. Returns (fold_index, model_path)."""
    res = run_training("bu_net", job["config_path"], job["run_dir"],
                       config_overrides={**job["fold_lists"],
                                         "models_dir": job["models_dir"],
                                         "epochs": job["epochs"]})
    if res.get("status") != "success":
        raise RuntimeError(f"teacher failed [fold{job['fi']}]: {res.get('error')}")
    return job["fi"], res["model_path"]


def _train_student_job(job: Dict[str, Any]):
    """Train + evaluate one (config, fold) Nano-U student. Returns (ci, fi, metrics)."""
    res = run_training("nano_u", job["config_path"], job["run_dir"],
                       config_overrides={**job["fold_lists"], **job["cfg"],
                                         "models_dir": job["models_dir"],
                                         "epochs": job["epochs"],
                                         "teacher_weights": job["teacher_weights"]})
    if res.get("status") != "success":
        raise RuntimeError(f"student failed [{job['tag']} fold{job['fi']}]: "
                           f"{res.get('error')}")
    config = load_config(job["config_path"])
    fm = evaluate_on_files(res["model_path"], job["fold_lists"]["cv_val_img"],
                           job["fold_lists"]["cv_val_mask"], config,
                           threshold=job["threshold"])
    return job["ci"], job["fi"], fm


def _run_jobs(fn, jobs: List[Dict[str, Any]], n_workers: int):
    """Run jobs sequentially (n_workers<=1) or across a spawn process pool.

    ``maxtasksperchild=1`` gives each job a fresh process, so a job's TF/CUDA
    GPU memory is fully released before the next starts — letting ``n_workers``
    full QAD pipelines coexist in VRAM without leak accumulation.
    """
    if n_workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        return pool.map(fn, jobs)


def run_cv(config_path: str, k: int, epochs: int, grid: List[Dict[str, Any]],
           output_dir: str, threshold: float = 0.5, seed: int = 37,
           fixed_teacher: Optional[str] = None,
           select_primary: str = "miou_mean", min_miou: float = 0.0,
           jobs: int = 1, teacher_jobs: int = 1) -> Dict[str, Any]:
    """Drive the full grouped-CV sweep. Returns the assembled table + winner.

    Teachers are trained **once per fold** (phase 1) and reused across every
    config that shares the fold — the teacher depends only on the leakage-safe
    fold split, not on the student hyperparameters, so this is exact, not an
    approximation. ``jobs`` runs that many student pipelines concurrently;
    ``teacher_jobs`` is separate because the 12.85M-param BU-Net teacher is far
    heavier than the 3.3k-param student — three concurrent teachers OOM a ~6 GB
    GPU, so teachers default to sequential.
    """
    config = load_config(config_path)
    img_files, mask_files = pool_train_val(config)
    mask_by_name = {os.path.basename(m): m for m in mask_files}
    img_files = [i for i in img_files if os.path.basename(i) in mask_by_name]
    groups = [sequence_group(i) for i in img_files]
    splits = grouped_kfold(groups, k=k, seed=seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Precompute the leakage-safe file lists for every fold.
    fold_lists_by_fold: Dict[int, Dict[str, Any]] = {}
    for fi, (tr_idx, va_idx) in enumerate(splits):
        tr_img = [img_files[j] for j in tr_idx]
        va_img = [img_files[j] for j in va_idx]
        fold_lists_by_fold[fi] = {
            "cv_train_img": tr_img,
            "cv_train_mask": [mask_by_name[os.path.basename(i)] for i in tr_img],
            "cv_val_img": va_img,
            "cv_val_mask": [mask_by_name[os.path.basename(i)] for i in va_img],
        }

    print(f"[CV] {len(img_files)} frames, {len(set(groups))} sequences, "
          f"{len(splits)} folds, {len(grid)} configs, jobs={jobs}")

    # ── Phase 1: one teacher per fold, cached & reused across all configs ──────
    if fixed_teacher is not None:
        print("[CV] using a single fixed teacher for every fold (leakage caveat)")
        teacher_of = {fi: fixed_teacher for fi in range(len(splits))}
    else:
        t_jobs = [{
            "config_path": config_path,
            "run_dir": str(out / "teachers" / f"fold{fi}"),
            "models_dir": str(out / "teachers" / f"fold{fi}" / "model"),
            "fold_lists": fold_lists_by_fold[fi],
            "epochs": epochs, "fi": fi,
        } for fi in range(len(splits))]
        print(f"[CV] training {len(t_jobs)} per-fold teachers "
              f"(vs {len(t_jobs) * len(grid)} before caching), "
              f"teacher_jobs={teacher_jobs}")
        teacher_of = dict(_run_jobs(_train_teacher_job, t_jobs,
                                    min(teacher_jobs, len(t_jobs))))

    # ── Phase 2: one student per (config, fold), reusing the fold's teacher ────
    s_jobs: List[Dict[str, Any]] = []
    tags: List[str] = []
    for ci, cfg in enumerate(grid):
        tag = (f"T{cfg['temperature']}_a{cfg['alpha']}_"
               f"{cfg['augment_regime']}_ce{int(cfg['ce_enabled'])}")
        if cfg.get("tversky_weight", 0.0) > 0.0:
            tag += f"_tv{cfg['tversky_weight']}"
        tags.append(tag)
        for fi in range(len(splits)):
            s_jobs.append({
                "config_path": config_path,
                "run_dir": str(out / "runs" / tag / f"fold{fi}"),
                "models_dir": str(out / "runs" / tag / f"fold{fi}" / "model"),
                "cfg": cfg, "fold_lists": fold_lists_by_fold[fi],
                "teacher_weights": teacher_of[fi], "epochs": epochs,
                "threshold": threshold, "ci": ci, "fi": fi, "tag": tag,
            })
    print(f"[CV] training {len(s_jobs)} student runs")
    student_results = _run_jobs(_train_student_job, s_jobs, min(jobs, len(s_jobs)))

    # ── Aggregate fold metrics per config ─────────────────────────────────────
    by_config: Dict[int, Dict[int, Dict[str, float]]] = {}
    for ci, fi, fm in student_results:
        by_config.setdefault(ci, {})[fi] = fm

    rows: List[Dict[str, Any]] = []
    for ci, cfg in enumerate(grid):
        fold_metrics = [by_config[ci][fi] for fi in sorted(by_config[ci])]
        agg = aggregate_folds(fold_metrics)
        rows.append(build_row(cfg, agg, fold_metrics))
        print(f"[{ci+1}/{len(grid)}] {tags[ci]}: "
              f"f0.5={agg['f0.5_mean']:.4f}±{agg['f0.5_std']:.4f} "
              f"miou={agg['miou_mean']:.4f}±{agg['miou_std']:.4f}")

    n_eligible = sum(1 for r in rows if r["miou_mean"] >= min_miou)
    if min_miou > 0.0 and n_eligible == 0:
        print(f"[CV] WARNING: no config reached the mIoU floor {min_miou:.3f}; "
              f"falling back to the highest-mIoU config (F0.5 floor infeasible).")
    elif min_miou > 0.0:
        print(f"[CV] {n_eligible}/{len(rows)} configs clear the mIoU floor "
              f"{min_miou:.3f}; selecting among those.")
    # mIoU-primary breaks ties on the F0.5 safety axis; any other primary breaks
    # ties on mIoU.
    select_secondary = "f0.5_mean" if select_primary == "miou_mean" else "miou_mean"
    best = select_best(rows, primary=select_primary, secondary=select_secondary,
                       min_miou=min_miou)
    _write_table(rows, out)
    print(f"\n[CV] BEST by {select_primary} ({select_secondary} tiebreak, "
          f"floor={min_miou:.3f}): "
          f"T={best['temperature']} alpha={best['alpha']} "
          f"regime={best['augment_regime']} ce={best['ce_enabled']} "
          f"→ f0.5={best['f0.5_mean']:.4f} miou={best['miou_mean']:.4f}")
    return {"rows": rows, "best": best}


def _write_table(rows: List[Dict[str, Any]], out: Path) -> None:
    """Persist the results table as both JSON and (via pandas) CSV."""
    (out / "cv_results.json").write_text(json.dumps(rows, indent=2))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "cv_results.csv", index=False)
    except Exception as e:  # pandas is in requirements; never fail the sweep on I/O
        print(f"[CV] CSV export skipped: {e}")
    print(f"[CV] wrote {out / 'cv_results.json'} and cv_results.csv")


def main():
    p = argparse.ArgumentParser(description="Leakage-safe grouped CV hyperparameter search")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--k", type=int, default=5, help="number of grouped folds")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--temperatures", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--regimes", nargs="+", default=["none", "geometric", "photometric", "full"],
                   choices=sorted(AUGMENT_REGIMES))
    p.add_argument("--ce", nargs="+", default=["on", "off"], choices=["on", "off"],
                   help="CE-loss ablation: 'on' keeps CE, 'off' sets alpha=1.0")
    p.add_argument("--tversky", type=float, nargs="+", default=[0.0],
                   help="conservative-loss ablation: student supervised loss is "
                        "(1-w)*BCE + w*Tversky(alpha_fp=0.7, beta_fn=0.3) for each "
                        "weight w. Default [0.0] = pure BCE (no sweep).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--fixed-teacher", default=None,
                   help="reuse one teacher .h5 for speed (logs a leakage caveat)")
    p.add_argument("--select", default="miou", choices=["f0.5", "f1", "f2", "miou"],
                   help="primary selection metric for Nano-U (default mIoU; F0.5 is "
                        "the tiebreak/safety axis). The teacher is always plain.")
    p.add_argument("--min-miou", type=float, default=0.0,
                   help="only relevant with a non-mIoU --select: restrict the choice "
                        "to configs whose mean mIoU clears this floor (e.g. 0.70); "
                        "falls back to best available if none qualify")
    p.add_argument("--jobs", type=int, default=1,
                   help="number of STUDENT pipelines to run concurrently (each in "
                        "its own process/GPU context). Set to how many tiny Nano-U "
                        "QAD pipelines fit in VRAM (e.g. 3). Default 1 = sequential.")
    p.add_argument("--teacher-jobs", type=int, default=1,
                   help="concurrent BU-Net teacher trainings. The teacher is ~12.85M "
                        "params and nearly fills a 6 GB GPU, so keep this low (default "
                        "1 = sequential); raise only on a larger GPU.")
    p.add_argument("--output", default=None, help="defaults to <results_dir>/cv")
    args = p.parse_args()

    if args.fixed_teacher:
        print("[CV] WARNING: --fixed-teacher reuses one teacher across folds; "
              "the teacher may have seen some folds' validation frames (leakage).")

    config = load_config(args.config)
    output = args.output or os.path.join(
        config.get("data", {}).get("paths", {}).get("results_dir", "results"), "cv")
    ce_options = [c == "on" for c in args.ce]
    grid = expand_grid(args.temperatures, args.alphas, args.regimes, ce_options,
                       args.tversky)

    run_cv(args.config, k=args.k, epochs=args.epochs, grid=grid, output_dir=output,
           threshold=args.threshold, seed=args.seed, fixed_teacher=args.fixed_teacher,
           select_primary=f"{args.select}_mean", min_miou=args.min_miou,
           jobs=args.jobs, teacher_jobs=args.teacher_jobs)


if __name__ == "__main__":
    main()
