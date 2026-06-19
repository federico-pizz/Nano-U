import os
import argparse
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

# Allow running the script directly (python src/evaluate.py)
# If executed directly, add project root so imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local utilities
from src.models import PadToMatch
from src.utils import get_project_root, BinaryIoU
from src.data import make_dataset, sorted_by_frame, sequence_group
from src.utils.config import load_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EPS = 1e-7


def sigmoid(x):
    return tf.math.sigmoid(x)


def dice_coef(y_true, y_pred, threshold=None):
    # y_pred may be logits or probabilities; assume probabilities
    if threshold is not None:
        y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * inter + EPS) / (denom + EPS)


def bce_loss_from_logits(y_true, logits):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(y_true, logits)


def focal_loss_from_logits(y_true, logits, gamma=2.0, alpha=0.25):
    """Focal loss with correct asymmetric alpha weighting.
    
    alpha weights the positive class; (1 - alpha) weights the negative class.
    This provides class-balance control, not just a loss scale.
    """
    prob = tf.math.sigmoid(logits)
    y_true = tf.cast(y_true, tf.float32)
    pt = tf.where(tf.equal(y_true, 1.0), prob, 1.0 - prob)
    # alpha_t: alpha for positives, (1-alpha) for negatives
    alpha_t = tf.where(tf.equal(y_true, 1.0),
                       tf.fill(tf.shape(y_true), alpha),
                       tf.fill(tf.shape(y_true), 1.0 - alpha))
    loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + EPS)
    return tf.reduce_mean(loss)


def fbeta_score(precision, recall, beta=1.0, eps=EPS):
    """F-beta = (1+b^2)·P·R / (b^2·P + R).

    beta>1 weights recall (e.g. F2), beta<1 weights precision (e.g. F0.5).
    For autonomous navigation the positive class is 'road/drivable', so a
    precision-weighted F0.5 (don't call it road unless sure) is the conservative
    selection criterion.
    """
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall + eps)


def _metrics_from_counts(tp, fp, fn, tn, eps=EPS):
    """Pixel-confusion counts → the metric dict reported everywhere."""
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou_fg + iou_bg)  # same formula as utils.metrics.BinaryIoU
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "miou": float(miou),
        "f0.5": float(fbeta_score(precision, recall, 0.5, eps)),
        "f1": float(fbeta_score(precision, recall, 1.0, eps)),
        "f2": float(fbeta_score(precision, recall, 2.0, eps)),
    }


def compute_segmentation_metrics(probs, masks, thresholds=None, groups=None,
                                 operating_threshold=0.5):
    """Rich, interpretable metrics from probability maps and binary masks.

    Args:
        probs:  float array (N, H, W[, 1]) of road/drivable probabilities in [0,1].
        masks:  array same leading shape; thresholded at 0.5 to a binary label.
        thresholds: thresholds for the sweep (default 0.05..0.95). The sweep gives
            a PR curve and lets the operating point be *chosen*, not assumed.
        groups: optional length-N sequence ids → per-sequence mean±std, exposing
            the variance the pooled headline hides.
        operating_threshold: decision threshold for the headline numbers — the
            conservativeness knob (higher = stricter 'road').

    Returns a JSON-serializable dict.
    """
    probs = np.asarray(probs, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    mask_bin = masks > 0.5
    n = probs.shape[0]
    axis = tuple(range(1, probs.ndim))

    # Per-image counts at the operating threshold → fast pooled/bootstrap/group sums.
    pred_bin = probs > operating_threshold
    tp_i = np.sum(mask_bin & pred_bin, axis=axis).astype(np.float64)
    fp_i = np.sum((~mask_bin) & pred_bin, axis=axis).astype(np.float64)
    fn_i = np.sum(mask_bin & (~pred_bin), axis=axis).astype(np.float64)
    tn_i = np.sum((~mask_bin) & (~pred_bin), axis=axis).astype(np.float64)

    res = _metrics_from_counts(tp_i.sum(), fp_i.sum(), fn_i.sum(), tn_i.sum())
    res["threshold"] = float(operating_threshold)
    res["n_images"] = int(n)

    # Threshold sweep (PR curve + metric-vs-threshold).
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    sweep = {"thresholds": [float(t) for t in thresholds],
             "precision": [], "recall": [], "f0.5": [], "f1": [], "f2": [], "miou": []}
    for t in thresholds:
        pb = probs > t
        m = _metrics_from_counts(
            float(np.sum(mask_bin & pb)), float(np.sum((~mask_bin) & pb)),
            float(np.sum(mask_bin & (~pb))), float(np.sum((~mask_bin) & (~pb))),
        )
        for k in ("precision", "recall", "f0.5", "f1", "f2", "miou"):
            sweep[k].append(m[k])
    res["threshold_sweep"] = sweep

    # Per-sequence breakdown (leakage-relevant: variance across scenes).
    if groups is not None:
        groups = list(groups)
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != n_images {n}")
        g_arr = np.array(groups)
        per = {"groups": [], "miou": [], "f0.5": []}
        for g in sorted(set(groups)):
            idx = np.where(g_arr == g)[0]
            m = _metrics_from_counts(tp_i[idx].sum(), fp_i[idx].sum(),
                                     fn_i[idx].sum(), tn_i[idx].sum())
            per["groups"].append(str(g))
            per["miou"].append(m["miou"])
            per["f0.5"].append(m["f0.5"])
        per["n_groups"] = len(per["groups"])
        for k in ("miou", "f0.5"):
            arr = np.array(per[k], dtype=np.float64)
            per[f"{k}_mean"] = float(arr.mean())
            per[f"{k}_std"] = float(arr.std())
        res["per_sequence"] = per

    return res


def run_inference(model_or_path, dataset, custom_objects=None):
    """Run a Keras model or an INT8 TFLite file over a tf.data.Dataset.

    Accepts a filesystem path (``.tflite`` / ``.h5`` / ``.keras``) or an already
    loaded Keras model. Returns ``(probs, masks, imgs)`` as float32 numpy arrays
    with probabilities in [0, 1] (sigmoid applied when the model emits logits).

    This is the single forward path shared by ``evaluate_and_plot`` and the CV
    harness, so device-format (INT8 TFLite) and float models are scored
    identically.
    """
    interpreter = None
    model = None
    input_details = output_details = None
    co = custom_objects or {"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}

    if isinstance(model_or_path, str):
        is_tflite = model_or_path.lower().endswith(".tflite")
        if is_tflite:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        module="tensorflow.lite")
                interpreter = tf.lite.Interpreter(model_path=model_or_path)
            try:
                interpreter.allocate_tensors()
            except RuntimeError:
                # SELECT_TF_OPS / Flex: fall back to a Keras sibling if present.
                base = os.path.splitext(model_or_path)[0]
                found = next((base + e for e in (".keras", ".h5", ".hdf5")
                             if os.path.exists(base + e)), None)
                if not found:
                    raise
                with tfmot.quantization.keras.quantize_scope():
                    with keras.utils.custom_object_scope(co):
                        model = keras.models.load_model(found, compile=False)
                interpreter = None
        else:
            with tfmot.quantization.keras.quantize_scope():
                with keras.utils.custom_object_scope(co):
                    model = keras.models.load_model(model_or_path, compile=False)
    else:
        model = model_or_path  # already-loaded Keras model

    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        in_det, out_det = input_details[0], output_details[0]
        in_dtype, out_dtype = in_det["dtype"], out_det["dtype"]
        q_scale, q_zero = (None, None)
        if in_det.get("quantization", (0.0, 0)) != (0.0, 0):
            q_scale, q_zero = in_det["quantization"]
        out_qs, out_qz = (None, None)
        if out_det.get("quantization", (0.0, 0)) != (0.0, 0):
            out_qs, out_qz = out_det["quantization"]

    def _forward(batch_imgs):
        """Return raw model output (logits or probs) for a batch as numpy."""
        if interpreter is not None:
            outs = []
            for i in range(batch_imgs.shape[0]):
                inp = batch_imgs[i:i + 1].numpy()
                if in_dtype in (np.int8, np.uint8):
                    if q_scale is None:
                        raise RuntimeError("TFLite expects quantized input but "
                                           "no quantization params were found")
                    inp = np.clip(np.round(inp / q_scale) + q_zero,
                                  np.iinfo(in_dtype).min,
                                  np.iinfo(in_dtype).max).astype(in_dtype)
                else:
                    inp = inp.astype(in_dtype)
                interpreter.set_tensor(in_det["index"], inp)
                interpreter.invoke()
                o = interpreter.get_tensor(out_det["index"])
                if out_dtype in (np.int8, np.uint8) and out_qs is not None:
                    o = (o.astype(np.float32) - out_qz) * out_qs
                else:
                    o = o.astype(np.float32)
                outs.append(o)
            return np.concatenate(outs, axis=0)
        return np.asarray(model(batch_imgs, training=False), dtype=np.float32)

    # Probe once whether the model emits probabilities or logits.
    probe_imgs = next(iter(dataset))[0]
    probe = _forward(probe_imgs[0:1])
    is_prob = bool(probe.min() >= -1e-3 and probe.max() <= 1.0 + 1e-3)

    all_probs, all_masks, all_imgs = [], [], []
    for imgs, masks in dataset:
        raw = _forward(imgs)
        probs = raw if is_prob else (1.0 / (1.0 + np.exp(-raw)))
        all_probs.append(np.asarray(probs, dtype=np.float32))
        all_masks.append(masks.numpy())
        all_imgs.append(imgs.numpy())

    return (np.concatenate(all_probs, axis=0),
            np.concatenate(all_masks, axis=0),
            np.concatenate(all_imgs, axis=0))


def evaluate_and_plot(model_name, config_path, batch_size=8, threshold=0.5,
                      samples_to_plot=6, out_path=None, split="test",
                      prefer="tflite"):
    """Evaluate a model by name and write metric artefacts + plots.

    ``prefer`` controls which on-disk format is scored when several siblings
    (``.tflite`` / ``.h5`` / ``.keras``) exist for ``model_name``:

      - ``"tflite"`` (default): score the INT8 ``.tflite`` — the device format.
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
    split_cfg = config['data']['paths']['processed'][split]
    test_img_dir = resolve_path(split_cfg['img'])
    test_mask_dir = resolve_path(split_cfg['mask'])

    test_imgs = [os.path.join(test_img_dir, f) for f in sorted(os.listdir(test_img_dir)) if f.endswith('.png')]
    test_masks = [os.path.join(test_mask_dir, f) for f in sorted(os.listdir(test_mask_dir)) if f.endswith('.png')]

    # make_dataset pairs images to masks by basename, so no manual alignment here.
    test_imgs = sorted_by_frame(test_imgs)
    test_masks = sorted_by_frame(test_masks)

    if len(test_imgs) == 0:
        raise RuntimeError(f'No images found for split={split} in {test_img_dir}')

    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])

    test_ds = make_dataset(
        test_imgs, test_masks,
        batch_size=batch_size, shuffle=False, augment=False,
        target_size=(input_shape[0], input_shape[1]),
        mean=mean, std=std
    )

    # Resolve model by name. ``prefer`` decides whether the INT8 .tflite or the
    # FP32 Keras file wins when both exist (see docstring).
    if prefer not in ("tflite", "float"):
        raise ValueError(f"prefer must be 'tflite'|'float', got {prefer!r}")
    models_dir = resolve_path(config['data']['paths']['models_dir'])
    tflite_c = [os.path.join(models_dir, f"{model_name}.tflite")]
    float_c = [os.path.join(models_dir, f"{model_name}.h5"),
               os.path.join(models_dir, f"{model_name}.keras")]
    candidates = (tflite_c + float_c) if prefer == "tflite" else (float_c + tflite_c)

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        raise FileNotFoundError(f'Model not found. Checked: {candidates}')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    print(f'[eval] split={split}  model={model_path}  images={len(test_imgs)}')

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
        all_probs, all_masks, groups=groups, operating_threshold=threshold,
    )

    # Backward-compatible loss metrics (bce/focal) for existing consumers.
    _logits = np.log(all_probs + EPS) - np.log(1.0 - all_probs + EPS)
    _masks_t = tf.convert_to_tensor(all_masks, dtype=tf.float32)
    _logits_t = tf.convert_to_tensor(_logits, dtype=tf.float32)
    results['bce'] = float(bce_loss_from_logits(_masks_t, _logits_t).numpy())
    results['focal'] = float(focal_loss_from_logits(_masks_t, _logits_t).numpy())

    print(f'Evaluation results (operating threshold = {threshold}):')
    for k in ('miou', 'dice', 'precision', 'recall', 'f0.5', 'f1', 'f2', 'bce', 'focal'):
        print(f'  {k}: {results[k]:.4f}')
    if 'per_sequence' in results:
        ps = results['per_sequence']
        print(f"  per-sequence miou: {ps['miou_mean']:.4f} ± {ps['miou_std']:.4f} "
              f"over {ps['n_groups']} sequences")

    # ── Output paths ──
    if out_path is None:
        config_results_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results")
        results_dir = os.path.join(root_dir, config_results_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f'eval_predictions_{split}.png')
    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    preds_bin = (all_probs > threshold).astype(np.uint8)

    # ── Qualitative panels: input / GT / prob / pred ──
    n_total = all_probs.shape[0]
    samples = max(1, min(samples_to_plot, n_total))
    sample_indices = np.linspace(0, n_total - 1, samples, dtype=int)
    fig, axes = plt.subplots(samples, 4, figsize=(12, 3 * samples))
    for idx, i in enumerate(sample_indices):
        img = np.clip(all_imgs[i, ...] * std_arr + mean_arr, 0.0, 1.0)
        row = axes[idx] if samples > 1 else axes
        row[0].imshow(img); row[0].set_title('Input'); row[0].axis('off')
        row[1].imshow(all_masks[i, ..., 0], cmap='gray'); row[1].set_title('Ground Truth'); row[1].axis('off')
        im = row[2].imshow(all_probs[i, ..., 0], cmap='viridis', vmin=0, vmax=1)
        row[2].set_title('Predicted Prob'); row[2].axis('off')
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.04)
        row[3].imshow(preds_bin[i, ..., 0], cmap='gray')
        row[3].set_title(f'Pred (thr={threshold})'); row[3].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Qualitative plot saved to {out_path}')

    # ── Statistical report: PR curve, metric-vs-threshold, and (only when
    # sequence groups exist) a per-sequence mIoU variance panel ──
    sweep = results['threshold_sweep']
    has_seq = 'per_sequence' in results
    ncols = 3 if has_seq else 2
    fig2, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    ax[0].plot(sweep['recall'], sweep['precision'], marker='o', ms=3)
    ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision')
    ax[0].set_title('Precision–Recall (road class)')
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].grid(True, alpha=0.3)
    for name in ('precision', 'recall', 'f0.5', 'f1', 'miou'):
        ax[1].plot(sweep['thresholds'], sweep[name], label=name)
    ax[1].axvline(threshold, color='k', ls='--', alpha=0.6, label=f'op={threshold}')
    ax[1].set_xlabel('Decision threshold'); ax[1].set_ylabel('Score')
    ax[1].set_title('Metric vs threshold (conservativeness knob)')
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    if has_seq:
        ps = results['per_sequence']
        x = np.arange(ps['n_groups'])
        ax[2].bar(x, ps['miou'])
        ax[2].axhline(ps['miou_mean'], color='r', ls='--', label=f"mean={ps['miou_mean']:.3f}")
        ax[2].set_xticks(x); ax[2].set_xticklabels(ps['groups'], rotation=90, fontsize=7)
        ax[2].set_ylabel('mIoU'); ax[2].set_title('Per-sequence mIoU (variance)')
        ax[2].legend(fontsize=8)
    plt.tight_layout()
    report_png = os.path.join(out_dir, f'{base}_metrics.png')
    plt.savefig(report_png, dpi=150)
    plt.close(fig2)
    print(f'Statistical report saved to {report_png}')

    # ── JSON reports ──
    import json as _json
    # Back-compat flat metrics (keys existing tools read) next to the plot.
    flat = {k: results[k] for k in
            ('bce', 'dice', 'focal', 'miou', 'precision', 'recall', 'f0.5', 'f1', 'f2')}
    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as _f:
        _json.dump(flat, _f, indent=2)
    # Format-tagged copy so tooling never confuses FP32 with INT8 metrics.
    fmt = 'int8' if model_path.lower().endswith('.tflite') else 'fp32'
    with open(os.path.join(out_dir, f'eval_results_{fmt}.json'), 'w') as _f:
        _json.dump(flat, _f, indent=2)
    print(f'Metrics saved to {os.path.join(out_dir, "eval_results.json")}')
    # Full rich report (sweep + per-sequence) — only when it carries the
    # per-sequence variance that the flat metrics don't already cover.
    if has_seq:
        report = dict(results); report['split'] = split; report['model'] = model_path
        with open(os.path.join(out_dir, f'{base}_report.json'), 'w') as _f:
            _json.dump(report, _f, indent=2)
        print(f'Full report saved to {base}_report.json')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and plot predictions')
    parser.add_argument('model', nargs='?', default='nano_u', help='Model to evaluate (.h5 or .tflite basename without extension)')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help="Split to evaluate (default test; val is what selection may look at)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Operating decision threshold for the headline metrics')
    args = parser.parse_args()

    evaluate_and_plot(args.model, args.config, batch_size=8, threshold=args.threshold,
                      samples_to_plot=6, split=args.split)
