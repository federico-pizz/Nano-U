"""
NAS-enabled training entrypoint - simplified wrapper.

This script is now a thin wrapper around src/train.py with --enable-nas flag.
The callback-based NAS monitoring approach eliminates the need for separate
model wrappers and complex layer introspection logic.

Usage:
    python src/train_with_nas.py --model nano_u --epochs 50
    
This is equivalent to:
    python src/train.py --model nano_u --epochs 50 --enable-nas

For backward compatibility, this script accepts all the same arguments as train.py
and automatically enables NAS monitoring.
"""

import sys
import os

# Ensure project root is importable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train with NAS monitoring enabled (wrapper around train.py)"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="nano_u", choices=["nano_u", "bu_net"], help="Model to train") 
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config (optional)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher-weights", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    
    # NAS-specific arguments (automatically enabled)
    parser.add_argument("--nas-log-dir", type=str, default=None, help="Directory for NAS TensorBoard logs")
    parser.add_argument("--nas-csv-path", type=str, default=None, help="Path for NAS metrics CSV output")
    parser.add_argument("--nas-log-freq", type=str, default="epoch", choices=["epoch", "batch"], 
                        help="NAS logging frequency")
    parser.add_argument("--nas-batch-freq", type=int, default=10, 
                        help="Batch frequency for NAS monitoring (when log_freq=batch)")
    
    args = parser.parse_args()

    print("=" * 60)
    print("NAS-ENABLED TRAINING (using callback-based monitoring)")
    print("=" * 60)
    
    # Call train() with NAS monitoring automatically enabled
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        distill=args.distill,
        teacher_weights=args.teacher_weights,
        alpha=args.alpha,
        temperature=args.temperature,
        augment=not args.no_augment,
        config_path=args.config,
        # NAS monitoring is always enabled when using this script
        enable_nas_monitoring=True,
        nas_log_dir=args.nas_log_dir,
        nas_csv_path=args.nas_csv_path,
        nas_log_freq=args.nas_log_freq,
        nas_monitor_batch_freq=args.nas_batch_freq
    )
    
    print("\n" + "=" * 60)
    print("NAS monitoring complete. Check logs for metrics.")
    print("=" * 60)
