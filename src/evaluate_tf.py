import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow running the script directly (python src/evaluate_tf.py)
# If executed directly, add project root so imports from `src` work.
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local utilities
from src.utils import get_project_root
from src.utils.data_tf import make_dataset
from src.utils.metrics_tf import BinaryIoU
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


def evaluate_and_plot(model_path, config_path, batch_size=8, threshold=0.5, samples_to_plot=6, out_path=None):
    config = load_config(config_path)
    root_dir = str(get_project_root())

    def resolve_path(p):
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    test_img_dir = resolve_path(config['data']['paths']['processed']['test']['img'])
    test_mask_dir = resolve_path(config['data']['paths']['processed']['test']['mask'])

    test_imgs = [os.path.join(test_img_dir, f) for f in sorted(os.listdir(test_img_dir)) if f.endswith('.png')]
    test_masks = [os.path.join(test_mask_dir, f) for f in sorted(os.listdir(test_mask_dir)) if f.endswith('.png')]

    if len(test_imgs) == 0:
        raise RuntimeError('No test images found')

    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']

    test_ds = make_dataset(test_imgs, test_masks, batch_size=batch_size, shuffle=False, augment=False, mean=mean, std=std)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

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
        logits = model(imgs, training=False)
        probs = sigmoid(logits)

        bce = bce_loss_from_logits(masks, logits).numpy()
        d = dice_coef(masks, probs, threshold=None).numpy()
        f = focal_loss_from_logits(masks, logits).numpy()

        preds_bin = (probs.numpy() > threshold).astype(np.uint8)

        mean_bce.update_state(bce)
        mean_dice.update_state(d)
        mean_focal.update_state(f)

        iou_metric.update_state(masks, logits)
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
        models_dir = resolve_path(config['data']['paths']['models_dir'])
        out_path = os.path.join(models_dir, 'eval_predictions.png')

    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    evaluate_and_plot(args.model_path, args.config, batch_size=args.batch_size, threshold=args.threshold, samples_to_plot=args.samples, out_path=args.out)
