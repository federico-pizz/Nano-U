"""Host-side evaluation, in steps a caller can use separately.

`score` turns probabilities and masks into the metrics dict, `plot` draws the
qualitative panels and the threshold report, `write_report` writes the one JSON
file. `evaluate` chains them for a saved model on a dataset split. The device
evaluator (`nanou.device.evaluate`) calls the same split loading, `score`,
`plot` and `write_report` on what the board sends back - which is what makes
host and device numbers comparable by construction.
"""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanou.config import ResolvedConfig, get_project_root, resolve
from nanou.data import make_dataset, sequence_group, sorted_by_frame
from nanou.metrics import (
    EPS,
    NumpyEncoder,
    bce_loss_from_logits,
    compute_segmentation_metrics,
    focal_loss_from_logits,
)
from nanou.predictor import load_predictor

SPLITS = ("train", "val", "test")
HEADLINE = ("miou", "dice", "precision", "recall", "f0.5", "f1", "f2", "bce", "focal")


def project_path(path) -> str:
    """A config path made absolute, relative paths being relative to the repository."""
    path = str(path)
    return path if os.path.isabs(path) else os.path.join(str(get_project_root()), path)


def split_files(config: ResolvedConfig, split: str = "test") -> tuple[list[str], list[str]]:
    """The (images, masks) of one split, in frame order - the order build.rs embeds them.

    Raises:
        ValueError: for a split other than train, val or test.
        RuntimeError: if the split has no images.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}")
    split_cfg = config.split_paths[split]
    img_dir, mask_dir = project_path(split_cfg["img"]), project_path(split_cfg["mask"])
    imgs = sorted_by_frame(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    )
    masks = sorted_by_frame(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
    )
    if not imgs:
        raise RuntimeError(f"No images found for split={split} in {img_dir}")
    return imgs, masks


def split_dataset(config: ResolvedConfig, imgs, masks, batch_size: int = 8):
    """The evaluation view of a split: normalised, unshuffled, unaugmented, in `imgs` order."""
    norm = config.normalization
    shape = config.input_shape
    return make_dataset(
        imgs,
        masks,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        target_size=(shape[0], shape[1]),
        mean=norm["mean"],
        std=norm["std"],
    )


def frame_groups(imgs, n_rows: int) -> list[str] | None:
    """The capture sequence of each row, or None if the rows no longer line up with `imgs`.

    make_dataset drops unpaired frames; labelling rows by position after that
    would assign them to the wrong sequence, so the breakdown is skipped instead.
    """
    groups = [sequence_group(p) for p in imgs]
    return groups if len(groups) == n_rows else None


def score(probs, masks, groups=None, threshold: float = 0.5) -> dict:
    """The metrics dict for probabilities in [0, 1] against binary masks.

    `compute_segmentation_metrics` plus the BCE and focal losses, which are
    computed from logits recovered from the probabilities so that every caller
    - a Keras model, a TFLite file, the board - is scored by the same formula.
    """
    results = compute_segmentation_metrics(
        probs, masks, groups=groups, operating_threshold=threshold
    )
    logits = np.log(probs + EPS) - np.log(1.0 - probs + EPS)
    masks_t = tf.convert_to_tensor(masks, dtype=tf.float32)
    logits_t = tf.convert_to_tensor(logits, dtype=tf.float32)
    results["bce"] = float(bce_loss_from_logits(masks_t, logits_t).numpy())
    results["focal"] = float(focal_loss_from_logits(masks_t, logits_t).numpy())
    return results


def print_headline(results: dict, title: str) -> None:
    """Print the headline metrics and, when present, the per-sequence spread."""
    print(f"{title} (operating threshold = {results['threshold']}):")
    for k in HEADLINE:
        print(f"  {k}: {results[k]:.4f}")
    if "per_sequence" in results:
        ps = results["per_sequence"]
        print(
            f"  per-sequence miou: {ps['miou_mean']:.4f} +/- {ps['miou_std']:.4f} "
            f"over {ps['n_groups']} sequences"
        )


def plot(
    probs,
    masks,
    imgs,
    results: dict,
    out_dir,
    name: str,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    samples: int = 6,
) -> list[str]:
    """Write ``<name>.png`` (input / GT / prob / pred panels) and ``<name>_metrics.png``.

    The second figure holds the PR curve, the metric-vs-threshold sweep and,
    when the results carry sequence groups, the per-sequence mIoU spread.

    Returns:
        The paths of the two figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    threshold = results["threshold"]
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    preds_bin = (probs > threshold).astype(np.uint8)

    n_total = probs.shape[0]
    samples = max(1, min(samples, n_total))
    sample_indices = np.linspace(0, n_total - 1, samples, dtype=int)
    fig, axes = plt.subplots(samples, 4, figsize=(12, 3 * samples))
    for idx, i in enumerate(sample_indices):
        img = np.clip(imgs[i, ...] * std_arr + mean_arr, 0.0, 1.0)
        row = axes[idx] if samples > 1 else axes
        row[0].imshow(img)
        row[0].set_title(f"Input {i}")
        row[0].axis("off")
        row[1].imshow(masks[i, ..., 0], cmap="gray")
        row[1].set_title("Ground Truth")
        row[1].axis("off")
        im = row[2].imshow(probs[i, ..., 0], cmap="viridis", vmin=0, vmax=1)
        row[2].set_title("Predicted Prob")
        row[2].axis("off")
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.04)
        row[3].imshow(preds_bin[i, ..., 0], cmap="gray")
        row[3].set_title(f"Pred (thr={threshold})")
        row[3].axis("off")
    plt.tight_layout()
    panels_png = os.path.join(out_dir, f"{name}.png")
    plt.savefig(panels_png, dpi=150)
    plt.close(fig)
    print(f"Qualitative plot saved to {panels_png}")

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
    for metric in ("precision", "recall", "f0.5", "f1", "miou"):
        ax[1].plot(sweep["thresholds"], sweep[metric], label=metric)
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
    report_png = os.path.join(out_dir, f"{name}_metrics.png")
    plt.savefig(report_png, dpi=150)
    plt.close(fig2)
    print(f"Statistical report saved to {report_png}")
    return [panels_png, report_png]


def write_report(results: dict, path, **context) -> str:
    """Write the metrics, and what produced them, to one JSON file.

    `context` (split, model, format, ...) goes first so the file says what it
    describes before the numbers start.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({**context, **results}, f, indent=2, cls=NumpyEncoder)
    print(f"Report saved to {path}")
    return path


def evaluate(
    model_path,
    config_path,
    split: str = "test",
    threshold: float = 0.5,
    out_dir=None,
    batch_size: int = 8,
    samples: int = 6,
) -> dict:
    """Score a saved model on one split and write its figures and report.

    The file decides the format: a ``.tflite`` is scored through the TFLite
    interpreter, a ``.h5``/``.keras`` as FP32 Keras. Outputs go to `out_dir`
    (default ``<results_dir>/<model stem>/``) as ``eval_<fmt>_<split>.png``,
    ``eval_<fmt>_<split>_metrics.png`` and ``eval_<fmt>_<split>.json``.

    Returns:
        The metrics dict.
    """
    config = resolve(path=config_path)
    imgs, masks = split_files(config, split)
    dataset = split_dataset(config, imgs, masks, batch_size)
    predictor = load_predictor(str(model_path))
    print(f"[eval] split={split}  model={model_path}  images={len(imgs)}")

    probs, gt, inputs = predictor.run(dataset)
    results = score(probs, gt, frame_groups(imgs, probs.shape[0]), threshold)
    print_headline(results, f"Evaluation results, {predictor.fmt}")

    if out_dir is None:
        results_dir = config.output_dir or "results"
        out_dir = project_path(os.path.join(results_dir, Path(model_path).stem))
    name = f"eval_{predictor.fmt}_{split}"
    norm = config.normalization
    plot(probs, gt, inputs, results, out_dir, name, norm["mean"], norm["std"], samples)
    write_report(
        results,
        os.path.join(out_dir, f"{name}.json"),
        split=split,
        model=str(model_path),
        format=predictor.fmt,
    )
    return results
