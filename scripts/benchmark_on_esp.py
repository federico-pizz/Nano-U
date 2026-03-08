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
import tf_keras as keras
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

def run_inference_on_device(repo_root, model_name="nano_u"):
    print(f"[1/3] Building inference binary for {model_name}...")
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    subprocess.run(["cargo", "build", "--release", "--bin", "inference"], cwd=repo_root / "esp_flash", check=True, env=env)
    
    print(f"\n[2/3] Flashing and running inference for {model_name}...")
    cmd = ["cargo", "run", "--release", "--bin", "inference"]
    
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, cwd=repo_root / "esp_flash", stdin=slave, stdout=slave, stderr=subprocess.STDOUT, close_fds=True, env=env)
    os.close(slave)
    
    buffer = b""
    start_time = time.time()
    capturing = False
    current_img = -1
    img_data = {}
    
    num_images = 0
    # Expected line length (IMG_W * 2) and row count (IMG_H)
    config = load_config(str(repo_root / 'config/config.yaml'))
    input_shape = config.get("data", {}).get("input_shape", [60, 80, 3])
    expected_row_len = input_shape[1] * 2
    expected_rows = input_shape[0]
    
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
                        # Dynamic hex string length checking
                        if len(line_str) == expected_row_len:
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

def evaluate_esp_outputs(img_data_dict, repo_root, model_name="nano_u", dataset_batch="botanic_garden", start_idx=0, end_idx=50, threshold=0.4, samples_to_plot=6):
    config_path = repo_root / 'config/config.yaml'
    config = load_config(str(config_path))
    
    # 1. Load actual test masks and inputs for the specific dataset batch
    if dataset_batch == "tinyagri":
        dataset_cfg = config.get("data", {}).get("paths", {}).get("secondary", {})
        test_cfg = dataset_cfg.get("test", {})
    else:
        data_paths = config.get("data", {}).get("paths", {})
        test_cfg = data_paths.get("processed", {}).get("test", {})
        
    test_img_dir = repo_root / test_cfg.get('img', '')
    test_mask_dir = repo_root / test_cfg.get('mask', '')
    
    test_imgs = sorted_by_frame([str(p) for p in test_img_dir.glob('*.png')])[:50]
    test_masks = sorted_by_frame([str(p) for p in test_mask_dir.glob('*.png')])[:50]
    
    mean = config['data']['normalization']['mean']
    std = config['data']['normalization']['std']
    test_ds = make_dataset(test_imgs, test_masks, batch_size=1, shuffle=False, augment=False, mean=mean, std=std)
    
    all_masks = []
    all_imgs = []
    for imgs, masks in test_ds:
        all_masks.append(masks.numpy()[0])
        all_imgs.append(imgs.numpy()[0])
        
    if not all_masks:
        print(f"Warning: No test images loaded for {dataset_batch}.")
        return None
        
    # 2. Load Dequantization Params from JSON (most reliable)
    params_path = repo_root / config['data']['paths']['models_dir'] / f"{model_name}_quant_params.json"
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
        out_qscale = params['output']['scale']
        out_qzero = params['output']['zero_point']
    else:
        out_qscale, out_qzero = 1.0, 0
        
    # 3. Process ESP Outputs
    mean_bce = keras.metrics.Mean(name='bce')
    mean_dice = keras.metrics.Mean(name='dice')
    mean_focal = keras.metrics.Mean(name='focal')
    iou_metric = BinaryIoU(threshold=threshold, name='binary_iou')
    precision = keras.metrics.Precision(name='precision')
    recall = keras.metrics.Recall(name='recall')
    
    all_probs = []
    all_preds_bin = []
    
    # We only evaluate the images in the specified range
    batch_img_indices = range(start_idx, min(end_idx, len(img_data_dict)))
    num_to_eval = len(batch_img_indices)
    
    print(f"\nEvaluating batch '{dataset_batch}' ({num_to_eval} images) for {model_name}...")
    
    expected_h = config['data']['input_shape'][0]
    if params_path.exists():
        with open(params_path, 'r') as f:
            p_data = json.load(f)
            expected_h = p_data['output']['shape'][1]

    for i_in_batch, i_global in enumerate(batch_img_indices):
        raw_rows = img_data_dict[i_global]
        if len(raw_rows) != expected_h:
            print(f"Warning: Image {i_global} has incomplete data, skipping.")
            continue
            
        out_i8 = np.stack(raw_rows, axis=0) 
        out_i8 = np.expand_dims(np.expand_dims(out_i8, axis=0), axis=-1)
        
        logits_f32 = (out_i8.astype(np.float32) - out_qzero) * out_qscale
        logits = tf.convert_to_tensor(logits_f32, dtype=tf.float32)
        probs = sigmoid(logits)
            
        masks_t = tf.convert_to_tensor(np.expand_dims(all_masks[i_in_batch], 0), dtype=tf.float32)
        
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
        
    if not all_probs:
        return None

    all_probs = np.stack(all_probs, axis=0)
    all_preds_bin = np.stack(all_preds_bin, axis=0)
    all_masks = np.stack(all_masks, axis=0)
    all_imgs = np.stack(all_imgs, axis=0)
    
    results = {
        'bce': float(mean_bce.result().numpy()),
        'dice': float(mean_dice.result().numpy()),
        'focal': float(mean_focal.result().numpy()),
        'iou': float(iou_metric.result().numpy()),
        'precision': float(precision.result().numpy()),
        'recall': float(recall.result().numpy()),
    }
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'] + EPS)
    
    print(f'Results for {dataset_batch}:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
        
    # Plotting
    samples = min(samples_to_plot, len(all_probs))
    sample_indices = np.linspace(0, len(all_probs) - 1, samples, dtype=int)
    fig, axes = plt.subplots(samples, 4, figsize=(12, 3 * samples))
    mean_arr, std_arr = np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

    for idx, i in enumerate(sample_indices):
        img_orig = np.clip((all_imgs[i] * std_arr) + mean_arr, 0.0, 1.0)
        mask = all_masks[i, ..., 0]
        prob = all_probs[i, ..., 0]
        pred_bin = all_preds_bin[i, ..., 0]

        ax0 = axes[idx, 0] if samples > 1 else axes[0]
        ax1 = axes[idx, 1] if samples > 1 else axes[1]
        ax2 = axes[idx, 2] if samples > 1 else axes[2]
        ax3 = axes[idx, 3] if samples > 1 else axes[3]

        ax0.imshow(img_orig)
        ax0.set_title(f'{dataset_batch} Img {i}')
        ax0.axis('off')
        ax1.imshow(mask, cmap='gray'); ax1.set_title('GT'); ax1.axis('off')
        ax2.imshow(prob, cmap='viridis', vmin=0, vmax=1); ax2.set_title('Prob'); ax2.axis('off')
        ax3.imshow(pred_bin, cmap='gray'); ax3.set_title('Pred'); ax3.axis('off')

    plt.tight_layout()
    out_dir = repo_root / 'results' / f'{model_name}_rust'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'esp32_hw_eval_{dataset_batch}.png'
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved to {out_path}')
    
    with open(out_dir / f'hw_metrics_{dataset_batch}.json', 'w') as mf:
        json.dump(results, mf, indent=2)
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark Nano-U model on ESP32 hardware')
    parser.add_argument('model', choices=['nano_u', 'nano_u2'], default='nano_u', nargs='?', help='Model to benchmark')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.resolve()
    print("==========================================")
    print(f"ESP32 Dual-Dataset Benchmarking Pipeline ({args.model})")
    print("==========================================")
    
    img_data = run_inference_on_device(repo_root, model_name=args.model)
    if not img_data:
        print("Failed to collect inference data from ESP32.")
        sys.exit(1)
        
    print("\n[3/3] Dequantizing ESP32 outputs and calculating metrics...")
    
    # Evaluate Botanic Garden (0-49)
    evaluate_esp_outputs(img_data, repo_root, model_name=args.model, dataset_batch="botanic_garden", start_idx=0, end_idx=50)
    
    # Evaluate TinyAgri (50-99)
    evaluate_esp_outputs(img_data, repo_root, model_name=args.model, dataset_batch="tinyagri", start_idx=50, end_idx=100)

if __name__ == '__main__':
    main()
