#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pty
import select
import time
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root so imports from `src` work.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_project_root, BinaryIoU
from src.data import make_dataset, sorted_by_frame
from src.utils.config import load_config
from src.evaluate import dice_coef, bce_loss_from_logits, focal_loss_from_logits, sigmoid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS = 1e-7

def run_inference_on_device(repo_root):
    print("[1/3] Building inference binary...")
    subprocess.run(["cargo", "build", "--release", "--bin", "inference"], cwd=repo_root / "esp_flash", check=True)
    
    print("\n[2/3] Flashing and running inference...")
    cmd = ["cargo", "run", "--release", "--bin", "inference"]
    
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, cwd=repo_root / "esp_flash", stdin=slave, stdout=slave, stderr=subprocess.STDOUT, close_fds=True)
    os.close(slave)
    
    buffer = b""
    start_time = time.time()
    capturing = False
    current_img = -1
    img_data = {}
    
    num_images = 0
    
    try:
        while True:
            if p.poll() is not None:
                break
            r, _, _ = select.select([master], [], [], 0.1)
            if master in r:
                try:
                    data = os.read(master, 1024)
                except OSError:
                    break
                if not data:
                    break
                
                sys.stdout.write(data.decode('utf-8', errors='replace'))
                sys.stdout.flush()
                
                buffer += data
                
                # Parse lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    line_str = line.decode('utf-8', errors='replace').strip()
                    
                    if line_str.startswith("BENCHMARK_START:"):
                        num_images = int(line_str.split(":")[1])
                        print(f"\nDevice announced {num_images} images.")
                        
                    elif line_str.startswith("IMG_OUTPUT_START:"):
                        current_img = int(line_str.split(":")[1])
                        img_data[current_img] = []
                        capturing = True
                        
                    elif line_str.startswith("IMG_OUTPUT_END:"):
                        capturing = False
                        
                    elif line_str == "BENCHMARK_DONE":
                        p.terminate()
                        return img_data
                        
                    elif capturing:
                        # This should be a hex string of length 160
                        if len(line_str) == 160:
                            # parse hex to int8
                            try:
                                row_bytes = bytes.fromhex(line_str)
                                # convert to signed int8
                                row_i8 = np.frombuffer(row_bytes, dtype=np.int8)
                                img_data[current_img].append(row_i8)
                            except ValueError:
                                pass
                                
        if time.time() - start_time > 300:
            print("\nTimeout reached!")
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if p.poll() is None:
            p.terminate()
            time.sleep(0.5)
            if p.poll() is None:
                p.kill()
        os.close(master)
        
    return img_data

def evaluate_esp_outputs(img_data_dict, repo_root, threshold=0.4, samples_to_plot=6):
    config_path = repo_root / 'config/config.yaml'
    config = load_config(str(config_path))
    
    # 1. Load actual test masks and inputs (for comparison and plotting)
    test_img_dir = repo_root / config['data']['paths']['processed']['test']['img']
    test_mask_dir = repo_root / config['data']['paths']['processed']['test']['mask']
    
    test_imgs = sorted_by_frame([str(p) for p in test_img_dir.glob('*.png')])
    test_masks = sorted_by_frame([str(p) for p in test_mask_dir.glob('*.png')])
    
    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']
    test_ds = make_dataset(test_imgs, test_masks, batch_size=1, shuffle=False, augment=False, mean=mean, std=std)
    
    all_masks = []
    all_imgs = []
    for imgs, masks in test_ds:
        all_masks.append(masks.numpy()[0])
        all_imgs.append(imgs.numpy()[0])
        
    if not all_masks:
        raise RuntimeError("No test images loaded.")
        
    # 2. Extract Dequantization Params from nano_u.tflite
    model_path = repo_root / config['data']['paths']['models_dir'] / "nano_u.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    out_det = interpreter.get_output_details()[0]
    out_qscale, out_qzero = out_det['quantization']
    if out_qscale == 0.0:
        print("Warning: TFLite model output is not quantized or missing scale.")
        out_qscale, out_qzero = 1.0, 0
        
    # 3. Process ESP Outputs
    mean_bce = tf.keras.metrics.Mean(name='bce')
    mean_dice = tf.keras.metrics.Mean(name='dice')
    mean_focal = tf.keras.metrics.Mean(name='focal')
    iou_metric = BinaryIoU(threshold=threshold, name='binary_iou')
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    
    all_probs = []
    all_preds_bin = []
    
    num_images = min(len(img_data_dict), len(all_masks))
    print(f"\nEvaluating {num_images} images from ESP32...")
    
    for i in range(num_images):
        raw_rows = img_data_dict[i]
        if len(raw_rows) != 60:
            print(f"Warning: Image {i} has incomplete data ({len(raw_rows)} rows), skipping.")
            continue
            
        # Shape: (60, 80)
        out_i8 = np.stack(raw_rows, axis=0) 
        # Add batch and channel back: (1, 60, 80, 1)
        out_i8 = np.expand_dims(np.expand_dims(out_i8, axis=0), axis=-1)
        
        # Dequantize
        out_f32 = (out_i8.astype(np.float32) - out_qzero) * out_qscale
        
        # Determine if Prob or Logits
        is_prob = out_f32.min() >= -1e-3 and out_f32.max() <= 1.0 + 1e-3
        if is_prob:
            probs = tf.convert_to_tensor(out_f32, dtype=tf.float32)
            logits = tf.math.log(probs + EPS) - tf.math.log(1.0 - probs + EPS)
        else:
            logits = tf.convert_to_tensor(out_f32, dtype=tf.float32)
            probs = sigmoid(logits)
            
        masks_t = tf.convert_to_tensor(np.expand_dims(all_masks[i], 0), dtype=tf.float32)
        
        bce = bce_loss_from_logits(masks_t, logits).numpy()
        d = dice_coef(masks_t, probs, threshold=None).numpy()
        f = focal_loss_from_logits(masks_t, logits).numpy()
        preds_bin = (probs.numpy() > threshold).astype(np.uint8)
        
        mean_bce.update_state(bce)
        mean_dice.update_state(d)
        mean_focal.update_state(f)
        iou_metric.update_state(masks_t, probs)
        precision.update_state(masks_t, preds_bin)
        recall.update_state(masks_t, preds_bin)
        
        all_probs.append(probs.numpy()[0])
        all_preds_bin.append(preds_bin[0])
        
    all_probs = np.stack(all_probs, axis=0)
    all_preds_bin = np.stack(all_preds_bin, axis=0)
    all_masks = np.stack(all_masks[:num_images], axis=0)
    all_imgs = np.stack(all_imgs[:num_images], axis=0)
    
    results = {
        'bce': float(mean_bce.result().numpy()),
        'dice': float(mean_dice.result().numpy()),
        'focal': float(mean_focal.result().numpy()),
        'iou': float(iou_metric.result().numpy()),
        'precision': float(precision.result().numpy()),
        'recall': float(recall.result().numpy()),
    }
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + EPS)
    
    print('\nESP32 On-Device Hardware Evaluation Results:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
        
    # Plotting
    samples = min(samples_to_plot, num_images)
    sample_indices = np.linspace(0, num_images - 1, samples, dtype=int)
    fig, axes = plt.subplots(samples, 4, figsize=(12, 3 * samples))
    mean_arr, std_arr = np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

    for idx, i in enumerate(sample_indices):
        img = np.clip((all_imgs[i] * std_arr) + mean_arr, 0.0, 1.0)
        mask = all_masks[i, ..., 0]
        prob = all_probs[i, ..., 0]
        pred_bin = all_preds_bin[i, ..., 0]

        ax0 = axes[idx, 0] if samples > 1 else axes[0]
        ax1 = axes[idx, 1] if samples > 1 else axes[1]
        ax2 = axes[idx, 2] if samples > 1 else axes[2]
        ax3 = axes[idx, 3] if samples > 1 else axes[3]

        ax0.imshow(img)
        ax0.set_title(f'ESP Image {i}')
        ax0.axis('off')

        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Ground Truth')
        ax1.axis('off')

        im = ax2.imshow(prob, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title('ESP Predicted Prob')
        ax2.axis('off')

        ax3.imshow(pred_bin, cmap='gray')
        ax3.set_title(f'ESP Pred (thr={threshold})')
        ax3.axis('off')

    plt.tight_layout()
    out_dir = repo_root / 'results' / 'nano_u_rust'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'esp32_hardware_eval.png'
    plt.savefig(out_path, dpi=150)
    print(f'\nHardware evaluation plot saved to {out_path}')
    
    with open(out_dir / 'hardware_metrics.json', 'w') as mf:
        json.dump(results, mf, indent=2)

def main():
    repo_root = Path(__file__).parent.parent.resolve()
    print("==========================================")
    print("ESP32 Hardware Benchmarking Pipeline")
    print("==========================================")
    
    img_data = run_inference_on_device(repo_root)
    if not img_data:
        print("Failed to collect inference data from ESP32.")
        sys.exit(1)
        
    print("\n[3/3] Dequantizing ESP32 outputs and calculating metrics...")
    evaluate_esp_outputs(img_data, repo_root, threshold=0.4)

if __name__ == '__main__':
    main()
