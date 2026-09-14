"""Host-side evaluation: score a named model on a split and write the report."""

import os

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanou.config import get_project_root, load_config
from nanou.data import make_dataset, sequence_group, sorted_by_frame
from nanou.metrics import (
    EPS,
    bce_loss_from_logits,
    compute_segmentation_metrics,
    focal_loss_from_logits,
)
from nanou.predictor import run_inference


def evaluate_and_plot(
    model_name,
    config_path,
    batch_size=8,
    threshold=0.5,
    samples_to_plot=6,
    out_path=None,
    split="test",
    prefer="tflite",
):
    """Evaluate a model by name and write metric artefacts + plots.

    ``prefer`` controls which on-disk format is scored when several siblings
    (``.tflite`` / ``.h5`` / ``.keras``) exist for ``model_name``:

      - ``"tflite"`` (default): score the INT8 ``.tflite`` - the device format.
      - ``"float"``:  score the FP32 Keras ``.h5`` / ``.keras`` instead.

    This matters because the resolver would otherwise *always* pick ``.tflite``,
    so the "float" metrics would silently be the quantised model. The flat
    metrics are written both to ``eval_results.json`` (back-compat) and to a
    format-tagged ``eval_results_{fp32,int8}.json`` so downstream tooling can
    tell the two apart.
    """
    config = load_config(config_path)
    root_dir = str(get_project_root())

    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    # Split-aware: evaluate on the held-out test set OR the validation set
    # (val is what model selection is allowed to look at; test is touched once).
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}")
    split_cfg = config["data"]["paths"]["processed"][split]
    test_img_dir = resolve_path(split_cfg["img"])
    test_mask_dir = resolve_path(split_cfg["mask"])

    test_imgs = [
        os.path.join(test_img_dir, f)
        for f in sorted(os.listdir(test_img_dir))
        if f.endswith(".png")
    ]
    test_masks = [
        os.path.join(test_mask_dir, f)
        for f in sorted(os.listdir(test_mask_dir))
        if f.endswith(".png")
    ]

    # make_dataset pairs images to masks by basename, so no manual alignment here.
    test_imgs = sorted_by_frame(test_imgs)
    test_masks = sorted_by_frame(test_masks)

    if len(test_imgs) == 0:
        raise RuntimeError(f"No images found for split={split} in {test_img_dir}")

    mean = config["data"]["normalization"]["mean"]
    std = config["data"]["normalization"]["std"]
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])

    test_ds = make_dataset(
        test_imgs,
        test_masks,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        target_size=(input_shape[0], input_shape[1]),
        mean=mean,
        std=std,
    )

    # Resolve model by name. ``prefer`` decides whether the INT8 .tflite or the
    # FP32 Keras file wins when both exist (see docstring).
    if prefer not in ("tflite", "float"):
        raise ValueError(f"prefer must be 'tflite'|'float', got {prefer!r}")
    models_dir = resolve_path(config["data"]["paths"]["models_dir"])
    tflite_c = [os.path.join(models_dir, f"{model_name}.tflite")]
    float_c = [
        os.path.join(models_dir, f"{model_name}.h5"),
        os.path.join(models_dir, f"{model_name}.keras"),
    ]
    candidates = (tflite_c + float_c) if prefer == "tflite" else (float_c + tflite_c)

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        raise FileNotFoundError(f"Model not found. Checked: {candidates}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[eval] split={split}  model={model_path}  images={len(test_imgs)}")

    # Single shared forward path (INT8 TFLite + Keras), probabilities in [0,1].
    all_probs, all_masks, all_imgs = run_inference(model_path, test_ds)

    # Per-sequence groups: make_dataset re-orders by frame and pairs by basename,
    # so the row order equals sorted_by_frame(test_imgs). Guard against any
    # mismatch (e.g. dropped unpaired frames) by skipping groups rather than
    # mislabeling rows.
    ordered = sorted_by_frame(test_imgs)
    groups = [sequence_group(p) for p in ordered]
    if len(groups) != all_probs.shape[0]:
        groups = None

    results = compute_segmentation_metrics(
        all_probs,
        all_masks,
        groups=groups,
        operating_threshold=threshold,
    )

    # Backward-compatible loss metrics (bce/focal) for existing consumers.
    _logits = np.log(all_probs + EPS) - np.log(1.0 - all_probs + EPS)
    _masks_t = tf.convert_to_tensor(all_masks, dtype=tf.float32)
    _logits_t = tf.convert_to_tensor(_logits, dtype=tf.float32)
    results["bce"] = float(bce_loss_from_logits(_masks_t, _logits_t).numpy())
    results["focal"] = float(focal_loss_from_logits(_masks_t, _logits_t).numpy())

    print(f"Evaluation results (operating threshold = {threshold}):")
    for k in ("miou", "dice", "precision", "recall", "f0.5", "f1", "f2", "bce", "focal"):
        print(f"  {k}: {results[k]:.4f}")
    if "per_sequence" in results:
        ps = results["per_sequence"]
        print(
            f"  per-sequence miou: {ps['miou_mean']:.4f} +/- {ps['miou_std']:.4f} "
            f"over {ps['n_groups']} sequences"
        )

    # -- Output paths --
    if out_path is None:
        config_results_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results")
        results_dir = os.path.join(root_dir, config_results_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"eval_predictions_{split}.png")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    preds_bin = (all_probs > threshold).astype(np.uint8)

    # -- Qualitative panels: input / GT / prob / pred --
    n_total = all_probs.shape[0]
    samples = max(1, min(samples_to_plot, n_total))
    sample_indices = np.linspace(0, n_total - 1, samples, dtype=int)
    fig, axes = plt.subplots(samples, 4, figsize=(12, 3 * samples))
    for idx, i in enumerate(sample_indices):
        img = np.clip(all_imgs[i, ...] * std_arr + mean_arr, 0.0, 1.0)
        row = axes[idx] if samples > 1 else axes
        row[0].imshow(img)
        row[0].set_title("Input")
        row[0].axis("off")
        row[1].imshow(all_masks[i, ..., 0], cmap="gray")
        row[1].set_title("Ground Truth")
        row[1].axis("off")
        im = row[2].imshow(all_probs[i, ..., 0], cmap="viridis", vmin=0, vmax=1)
        row[2].set_title("Predicted Prob")
        row[2].axis("off")
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.04)
        row[3].imshow(preds_bin[i, ..., 0], cmap="gray")
        row[3].set_title(f"Pred (thr={threshold})")
        row[3].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Qualitative plot saved to {out_path}")

    # -- Statistical report: PR curve, metric-vs-threshold, and (only when
    # sequence groups exist) a per-sequence mIoU variance panel --
    sweep = results["threshold_sweep"]
    has_seq = "per_sequence" in results
    ncols = 3 if has_seq else 2
    fig2, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    ax[0].plot(sweep["recall"], sweep["precision"], marker="o", ms=3)
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall (road class)")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].grid(True, alpha=0.3)
    for name in ("precision", "recall", "f0.5", "f1", "miou"):
        ax[1].plot(sweep["thresholds"], sweep[name], label=name)
    ax[1].axvline(threshold, color="k", ls="--", alpha=0.6, label=f"op={threshold}")
    ax[1].set_xlabel("Decision threshold")
    ax[1].set_ylabel("Score")
    ax[1].set_title("Metric vs threshold (conservativeness knob)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    if has_seq:
        ps = results["per_sequence"]
        x = np.arange(ps["n_groups"])
        ax[2].bar(x, ps["miou"])
        ax[2].axhline(ps["miou_mean"], color="r", ls="--", label=f"mean={ps['miou_mean']:.3f}")
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(ps["groups"], rotation=90, fontsize=7)
        ax[2].set_ylabel("mIoU")
        ax[2].set_title("Per-sequence mIoU (variance)")
        ax[2].legend(fontsize=8)
    plt.tight_layout()
    report_png = os.path.join(out_dir, f"{base}_metrics.png")
    plt.savefig(report_png, dpi=150)
    plt.close(fig2)
    print(f"Statistical report saved to {report_png}")

    # -- JSON reports --
    import json as _json

    # Back-compat flat metrics (keys existing tools read) next to the plot.
    flat = {
        k: results[k]
        for k in ("bce", "dice", "focal", "miou", "precision", "recall", "f0.5", "f1", "f2")
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as _f:
        _json.dump(flat, _f, indent=2)
    # Format-tagged copy so tooling never confuses FP32 with INT8 metrics.
    fmt = "int8" if model_path.lower().endswith(".tflite") else "fp32"
    with open(os.path.join(out_dir, f"eval_results_{fmt}.json"), "w") as _f:
        _json.dump(flat, _f, indent=2)
    print(f"Metrics saved to {os.path.join(out_dir, 'eval_results.json')}")
    # Full rich report (sweep + per-sequence) - only when it carries the
    # per-sequence variance that the flat metrics don't already cover.
    if has_seq:
        report = dict(results)
        report["split"] = split
        report["model"] = model_path
        with open(os.path.join(out_dir, f"{base}_report.json"), "w") as _f:
            _json.dump(report, _f, indent=2)
        print(f"Full report saved to {base}_report.json")

    return results
