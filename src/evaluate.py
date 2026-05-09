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
from src.data import make_dataset, sorted_by_frame
from src.utils.config import load_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EPS = 1e-7


def _align_masks_to_images(img_paths: list, mask_paths: list) -> list:
    """Return mask_paths reordered to align with img_paths by frame number.

    Falls back to positional pairing with a warning when frame numbers are absent.
    """
    import re

    def first_num(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    img_nums = [first_num(p) for p in img_paths]
    mask_nums = [first_num(p) for p in mask_paths]

    if any(n is not None for n in img_nums) and any(n is not None for n in mask_nums):
        if img_nums != mask_nums:
            mask_map = {n: p for n, p in zip(mask_nums, mask_paths) if n is not None}
            aligned = []
            missing = False
            fallback = list(mask_paths)
            for n in img_nums:
                if n in mask_map:
                    aligned.append(mask_map[n])
                else:
                    missing = True
                    aligned.append(fallback.pop(0) if fallback else None)
            print(f'Warning: reordered masks to match image frame numbers; missing matches: {missing}')
            return aligned
    else:
        imgs_b = [os.path.basename(p) for p in img_paths]
        masks_b = [os.path.basename(p) for p in mask_paths]
        mismatches = sum(1 for a, b in zip(imgs_b, masks_b) if a != b)
        if mismatches:
            print(f'Warning: {mismatches} filename mismatches between images and masks; proceeding with positional pairing.')

    return mask_paths


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


def evaluate_and_plot(model_name, config_path, batch_size=8, threshold=0.5, samples_to_plot=6, out_path=None):
    config = load_config(config_path)
    root_dir = str(get_project_root())

    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    test_img_dir = resolve_path(config['data']['paths']['processed']['test']['img'])
    test_mask_dir = resolve_path(config['data']['paths']['processed']['test']['mask'])

    test_imgs = [os.path.join(test_img_dir, f) for f in sorted(os.listdir(test_img_dir)) if f.endswith('.png')]
    test_masks = [os.path.join(test_mask_dir, f) for f in sorted(os.listdir(test_mask_dir)) if f.endswith('.png')]

    test_imgs = sorted_by_frame(test_imgs)
    test_masks = sorted_by_frame(test_masks)
    test_masks = _align_masks_to_images(test_imgs, test_masks)

    if len(test_imgs) == 0:
        raise RuntimeError('No test images found')

    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])

    test_ds = make_dataset(
        test_imgs, test_masks,
        batch_size=batch_size, shuffle=False, augment=False,
        target_size=(input_shape[0], input_shape[1]),
        mean=mean, std=std
    )

    # Resolve model by name (prefer .tflite, then .keras/.h5)
    models_dir = resolve_path(config['data']['paths']['models_dir'])
    candidates = [
        os.path.join(models_dir, f"{model_name}.tflite"),
        os.path.join(models_dir, f"{model_name}.h5"),
        os.path.join(models_dir, f"{model_name}.keras"),
    ]

    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        raise FileNotFoundError(f'Model not found. Checked: {candidates}')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    is_tflite = model_path.lower().endswith('.tflite')

    interpreter = None
    model = None
    if is_tflite:
        print(f'Loading TFLite model: {model_path}')
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")
            interpreter = tf.lite.Interpreter(model_path=model_path)
        try:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            # print('TFLite input details:', input_details)
            # print('TFLite output details:', output_details)
        except RuntimeError as e:
            # Common case: TFLite model contains SELECT_TF_OPS (Flex) and interpreter needs Flex delegate
            msg = str(e)
            print('TFLite interpreter failed to allocate tensors:', msg)
            print('Attempting to fall back to the original Keras model if available...')
            # Robustly derive candidate Keras filenames from the tflite path
            keras_candidate_base = os.path.splitext(model_path)[0]
            candidates = [keras_candidate_base + ext for ext in ('.keras', '.h5', '.hdf5')]
            found = None
            for cand in candidates:
                if os.path.exists(cand):
                    found = cand
                    break

            if found:
                print(f'Found Keras model fallback: {found}. Loading it instead for evaluation.')
                with tfmot.quantization.keras.quantize_scope():
                    with keras.utils.custom_object_scope({'BinaryIoU': BinaryIoU, 'PadToMatch': PadToMatch}):
                        model = keras.models.load_model(found, compile=False)
                interpreter = None
            else:
                raise RuntimeError(
                    msg + '\nThe TFLite model requires the Flex delegate (TF Select ops).\n'
                    'Either rebuild the TFLite without SELECT_TF_OPS, provide the Flex delegate to the interpreter,\n'
                    'or place a Keras model file (one of .keras/.h5/.hdf5) alongside the .tflite file to be used as fallback.'
                )
    else:
        print(f'Loading Keras model: {model_path}')
        with tfmot.quantization.keras.quantize_scope():
            with keras.utils.custom_object_scope({'BinaryIoU': BinaryIoU, 'PadToMatch': PadToMatch}):
                model = keras.models.load_model(model_path, compile=False)

    # Trackers
    mean_bce = keras.metrics.Mean(name='bce')
    mean_dice = keras.metrics.Mean(name='dice')
    mean_focal = keras.metrics.Mean(name='focal')
    iou_metric = BinaryIoU(threshold=threshold, name='binary_iou')
    precision = keras.metrics.Precision(name='precision')
    recall = keras.metrics.Recall(name='recall')

    all_probs = []
    all_preds_bin = []
    all_masks = []
    all_imgs = []

    print('Running evaluation on test set...')

    # -----------------------------------------------------------------------
    # Determine ONCE whether the model outputs probabilities or raw logits.
    # Doing this per-batch is unreliable: a batch of uncertain logit values
    # that happen to fall in [0, 1] would be wrongly treated as probabilities.
    # We probe a single sample before the main loop instead.
    # -----------------------------------------------------------------------
    def _run_single_sample(sample_img):
        """Run the model on one sample (shape [1, H, W, C]) and return a numpy array."""
        if is_tflite and interpreter is not None:
            in_det = input_details[0]
            in_dtype = in_det['dtype']
            inp = sample_img.numpy()
            q_sc, q_zp = (None, None)
            if 'quantization' in in_det and in_det['quantization'] != (0.0, 0):
                q_sc, q_zp = in_det['quantization']
            if (in_dtype == np.int8 or in_dtype == np.uint8) and q_sc is not None:
                inp = np.clip(np.round(inp / q_sc) + q_zp,
                              np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
            else:
                inp = inp.astype(in_dtype)
            interpreter.set_tensor(in_det['index'], inp)
            interpreter.invoke()
            out_det = output_details[0]
            out = interpreter.get_tensor(out_det['index'])
            out_dtype = out_det['dtype']
            out_qs, out_qz = (None, None)
            if 'quantization' in out_det and out_det['quantization'] != (0.0, 0):
                out_qs, out_qz = out_det['quantization']
            if (out_dtype == np.int8 or out_dtype == np.uint8) and out_qs is not None:
                out = (out.astype(np.float32) - out_qz) * out_qs
            else:
                out = out.astype(np.float32)
            return out
        else:
            return model(sample_img, training=False).numpy()

    _probe_batch = next(iter(test_ds))
    _probe_out = _run_single_sample(_probe_batch[0][0:1])
    is_prob = bool(_probe_out.min() >= -1e-3 and _probe_out.max() <= 1.0 + 1e-3)
    print(f'Output type: {"probabilities" if is_prob else "logits"} '
          f'(probe range [{_probe_out.min():.3f}, {_probe_out.max():.3f}])')

    # Cache TFLite quantization params outside the inner loop to avoid re-reading per batch
    if is_tflite and interpreter is not None:
        in_det = input_details[0]
        in_index = in_det['index']
        in_dtype = in_det['dtype']
        q_scale, q_zero = (None, None)
        if 'quantization' in in_det and in_det['quantization'] != (0.0, 0):
            q_scale, q_zero = in_det['quantization']
        out_det = output_details[0]
        out_dtype = out_det['dtype']
        out_qscale, out_qzero = (None, None)
        if 'quantization' in out_det and out_det['quantization'] != (0.0, 0):
            out_qscale, out_qzero = out_det['quantization']

    for imgs, masks in test_ds:
        # If we loaded a TFLite interpreter and it is usable, run it; otherwise use the Keras model
        if is_tflite and interpreter is not None:
            # TFLite interpreter may not accept dynamic batch sizes; run sample-wise to be safe
            batch_outs = []

            for i in range(imgs.shape[0]):
                inp = imgs[i:i+1].numpy()
                # quantize if needed for input
                if in_dtype == np.int8 or in_dtype == np.uint8:
                    if q_scale is None:
                        raise RuntimeError('TFLite model expects quantized input but no quantization params found')
                    inp_scaled = np.round(inp / q_scale) + q_zero
                    inp_q = np.clip(inp_scaled, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
                    interpreter.set_tensor(in_index, inp_q)
                else:
                    interpreter.set_tensor(in_index, inp.astype(in_dtype))
                interpreter.invoke()
                out = interpreter.get_tensor(out_det['index'])
                # Dequantize output if quantized
                if (out_dtype == np.int8 or out_dtype == np.uint8) and out_qscale is not None:
                    out = (out.astype(np.float32) - out_qzero) * out_qscale
                else:
                    out = out.astype(np.float32)
                batch_outs.append(out)
            outputs = np.concatenate(batch_outs, axis=0)
            # Ensure outputs are float32 numpy array before converting to tensor
            outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
        else:
            outputs = model(imgs, training=False)

        # Ensure we operate on a numpy float32 array for dtype-safe ops
        if isinstance(outputs, tf.Tensor):
            out_np = outputs.numpy()
        else:
            out_np = np.array(outputs)

        # is_prob is determined once before the loop — do not recompute here
        if is_prob:
            probs = tf.convert_to_tensor(out_np, dtype=tf.float32)
            # convert probabilities to logits for loss functions expecting logits
            logits = tf.math.log(probs + EPS) - tf.math.log(1.0 - probs + EPS)
        else:
            logits = tf.convert_to_tensor(out_np, dtype=tf.float32)
            probs = sigmoid(logits)

        # compute losses/metrics using logits where required
        bce = bce_loss_from_logits(masks, logits).numpy()
        # Binary Dice at the deployment threshold (not soft Dice)
        d = dice_coef(masks, probs, threshold=threshold).numpy()
        f = focal_loss_from_logits(masks, logits).numpy()

        preds_bin = (probs.numpy() > threshold).astype(np.uint8)

        mean_bce.update_state(bce)
        mean_dice.update_state(d)
        mean_focal.update_state(f)

        # IoU metric expects probabilities (it will threshold internally using the configured threshold)
        iou_metric.update_state(masks, probs)
        precision.update_state(masks, preds_bin)
        recall.update_state(masks, preds_bin)

        all_probs.append(probs.numpy())
        all_preds_bin.append(preds_bin)
        all_masks.append(masks.numpy())
        all_imgs.append(imgs.numpy())

    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds_bin = np.concatenate(all_preds_bin, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_imgs = np.concatenate(all_imgs, axis=0)

    results = {
        'bce': float(mean_bce.result().numpy()),
        'dice': float(mean_dice.result().numpy()),
        'focal': float(mean_focal.result().numpy()),
        'miou': float(iou_metric.result().numpy()),
        'precision': float(precision.result().numpy()),
        'recall': float(recall.result().numpy()),
    }
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + EPS)

    print('Evaluation results:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    # Plot samples — pick evenly spaced indices across the full test set
    n_total = all_probs.shape[0]
    samples = min(samples_to_plot, n_total)
    sample_indices = np.linspace(0, n_total - 1, samples, dtype=int)
    fig_rows = samples
    fig, axes = plt.subplots(fig_rows, 4, figsize=(12, 3 * fig_rows))

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    for idx, i in enumerate(sample_indices):
        # Undo normalization from dataset tensors natively in memory
        img = (all_imgs[i, ...] * std_arr) + mean_arr
        img = np.clip(img, 0.0, 1.0)
        
        mask = all_masks[i, ...]
        prob = all_probs[i, ..., 0]
        pred_bin = all_preds_bin[i, ..., 0]

        ax0 = axes[idx, 0] if fig_rows > 1 else axes[0]
        ax1 = axes[idx, 1] if fig_rows > 1 else axes[1]
        ax2 = axes[idx, 2] if fig_rows > 1 else axes[2]
        ax3 = axes[idx, 3] if fig_rows > 1 else axes[3]

        ax0.imshow(img)
        ax0.set_title('Input')
        ax0.axis('off')

        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Ground Truth')
        ax1.axis('off')

        im = ax2.imshow(prob, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title('Predicted Prob')
        ax2.axis('off')
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax3.imshow(pred_bin, cmap='gray')
        ax3.set_title(f'Pred (thr={threshold})')
        ax3.axis('off')

    plt.tight_layout()

    if out_path is None:
        config_results_dir = config.get("data", {}).get("paths", {}).get("results_dir", "results")
        results_dir = os.path.join(root_dir, config_results_dir, model_name)
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, 'eval_predictions.png')

    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')

    # Always save metrics to eval_results.json alongside the plot so that
    # summarize_results.py can compare float32 and INT8 on the same test set.
    eval_results_path = os.path.join(os.path.dirname(out_path), 'eval_results.json')
    import json as _json
    with open(eval_results_path, 'w') as _f:
        _json.dump(results, _f, indent=2)
    print(f'Metrics saved to {eval_results_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and plot predictions')
    parser.add_argument('model', nargs='?', default='nano_u', help='Model to evaluate (.h5 or .tflite basename without extension)')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    evaluate_and_plot(args.model, args.config, batch_size=8, threshold=0.5, samples_to_plot=6)
