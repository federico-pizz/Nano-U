import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow running the script directly (python src/evaluate.py)
# If executed directly, add project root so imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local utilities
from src.utils import get_project_root
from src.utils.data import make_dataset, sorted_by_frame
from src.utils.metrics import BinaryIoU
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
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(y_true, logits)


def focal_loss_from_logits(y_true, logits, gamma=2.0, alpha=0.25):
    # Implementation adapted for logits input
    prob = tf.math.sigmoid(logits)
    y_true = tf.cast(y_true, tf.float32)
    pt = tf.where(tf.equal(y_true, 1.0), prob, 1.0 - prob)
    loss = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + EPS)
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

    # Ensure pairs are aligned by filename sorting or frame index
    test_imgs = sorted_by_frame(test_imgs)
    test_masks = sorted_by_frame(test_masks)

    # Verify pairing: extract first numeric token from filenames and align masks to image order if necessary
    import re
    def first_num(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    img_nums = [first_num(p) for p in test_imgs]
    mask_nums = [first_num(p) for p in test_masks]

    if any(n is not None for n in img_nums) and any(n is not None for n in mask_nums):
        if img_nums != mask_nums:
            mask_map = {n: p for n, p in zip(mask_nums, test_masks) if n is not None}
            new_masks = []
            missing = False
            tmp_masks = list(test_masks)
            for n in img_nums:
                if n in mask_map:
                    new_masks.append(mask_map[n])
                else:
                    missing = True
                    # fallback: take next available mask
                    new_masks.append(tmp_masks.pop(0) if tmp_masks else None)
            test_masks = new_masks
            print(f'Warning: Reordered masks to match image frame numbers; missing matches: {missing}')
    else:
        # fallback: check simple basename alignment and warn if different
        imgs_b = [os.path.basename(p) for p in test_imgs]
        masks_b = [os.path.basename(p) for p in test_masks]
        if len(imgs_b) == len(masks_b):
            mismatches = sum(1 for a, b in zip(imgs_b, masks_b) if a != b)
            if mismatches:
                print(f'Warning: {mismatches} filename mismatches between images and masks; proceeding with positional pairing.')

    if len(test_imgs) == 0:
        raise RuntimeError('No test images found')

    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']

    test_ds = make_dataset(test_imgs, test_masks, batch_size=batch_size, shuffle=False, augment=False, mean=mean, std=std)

    # Resolve model by name (prefer .tflite, then .keras/.h5)
    models_dir = resolve_path(config['data']['paths']['models_dir'])
    candidates = [
        os.path.join(models_dir, f"{model_name}.tflite"),
        os.path.join(models_dir, f"{model_name}.keras"),
        os.path.join(models_dir, f"{model_name}.h5"),
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
        interpreter = tf.lite.Interpreter(model_path=model_path)
        try:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print('TFLite input details:', input_details)
            print('TFLite output details:', output_details)
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
                model = keras.models.load_model(found, compile=False)
                model.summary()
                interpreter = None
            else:
                raise RuntimeError(
                    msg + '\nThe TFLite model requires the Flex delegate (TF Select ops).\n'
                    'Either rebuild the TFLite without SELECT_TF_OPS, provide the Flex delegate to the interpreter,\n'
                    'or place a Keras model file (one of .keras/.h5/.hdf5) alongside the .tflite file to be used as fallback.'
                )
    else:
        model = keras.models.load_model(model_path, compile=False)
        model.summary()

    # Trackers
    mean_bce = tf.keras.metrics.Mean(name='bce')
    mean_dice = tf.keras.metrics.Mean(name='dice')
    mean_focal = tf.keras.metrics.Mean(name='focal')
    iou_metric = BinaryIoU(threshold=threshold, name='binary_iou')
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')

    all_probs = []
    all_preds_bin = []
    all_masks = []

    print('Running evaluation on test set...')

    for imgs, masks in test_ds:
        # If we loaded a TFLite interpreter and it is usable, run it; otherwise use the Keras model
        if is_tflite and interpreter is not None:
            # TFLite interpreter may not accept dynamic batch sizes; run sample-wise to be safe
            batch_outs = []
            in_det = input_details[0]
            in_index = in_det['index']
            in_dtype = in_det['dtype']
            q_scale, q_zero = (None, None)
            if 'quantization' in in_det and in_det['quantization'] != (0.0, 0):
                q_scale, q_zero = in_det['quantization']

            for i in range(imgs.shape[0]):
                inp = imgs[i:i+1].numpy()
                # quantize if needed
                if in_dtype == np.int8 or in_dtype == np.uint8:
                    if q_scale is None:
                        raise RuntimeError('TFLite model expects quantized input but no quantization params found')
                    inp_q = (inp / q_scale + q_zero).astype(in_dtype)
                    interpreter.set_tensor(in_index, inp_q)
                else:
                    interpreter.set_tensor(in_index, inp.astype(in_dtype))
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])
                batch_outs.append(out)
            outputs = np.concatenate(batch_outs, axis=0)
            outputs = tf.convert_to_tensor(outputs)
        else:
            outputs = model(imgs, training=False)

        # Detect whether model output is probabilities (in [0,1]) or raw logits.
        out_np = outputs.numpy()
        is_prob = out_np.min() >= -1e-6 and out_np.max() <= 1.0 + 1e-6

        if is_prob:
            probs = tf.convert_to_tensor(outputs)
            # convert probabilities to logits for loss functions expecting logits
            logits = tf.math.log(probs + EPS) - tf.math.log(1.0 - probs + EPS)
        else:
            logits = tf.convert_to_tensor(outputs)
            probs = sigmoid(logits)

        # compute losses/metrics using logits where required
        bce = bce_loss_from_logits(masks, logits).numpy()
        d = dice_coef(masks, probs, threshold=None).numpy()
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

    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds_bin = np.concatenate(all_preds_bin, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    results = {
        'bce': float(mean_bce.result().numpy()),
        'dice': float(mean_dice.result().numpy()),
        'focal': float(mean_focal.result().numpy()),
        'iou': float(iou_metric.result().numpy()),
        'precision': float(precision.result().numpy()),
        'recall': float(recall.result().numpy()),
    }
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + EPS)

    print('Evaluation results:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    # Plot samples
    samples = min(samples_to_plot, all_probs.shape[0])
    fig_rows = samples
    fig, axes = plt.subplots(fig_rows, 4, figsize=(12, 3 * fig_rows))

    # Need to load original (un-normalized) images for visualization; reuse loader from make_dataset internal function
    # Simple approach: read with tensorflow image decode and undo normalization
    import cv2

    for i in range(samples):
        # read raw image and mask
        img = cv2.imread(test_imgs[i])[:, :, ::-1] / 255.0
        mask = cv2.imread(test_masks[i], cv2.IMREAD_GRAYSCALE) / 255.0
        prob = all_probs[i, ..., 0]
        pred_bin = all_preds_bin[i, ..., 0]

        ax0 = axes[i, 0] if fig_rows > 1 else axes[0]
        ax1 = axes[i, 1] if fig_rows > 1 else axes[1]
        ax2 = axes[i, 2] if fig_rows > 1 else axes[2]
        ax3 = axes[i, 3] if fig_rows > 1 else axes[3]

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
        results_dir = os.path.join(root_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, 'eval_predictions.png')

    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model-name', required=False, default='nano_u',
                        help='Model basename (without extension). Script will locate .tflite/.keras/.h5 in models dir')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    evaluate_and_plot(args.model_name, args.config, batch_size=args.batch_size, threshold=args.threshold, samples_to_plot=args.samples, out_path=args.out)
