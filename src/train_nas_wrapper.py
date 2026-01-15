#!/usr/bin/env python3
# Compatibility wrapper: thin CLI around src.train_with_nas.train
# Minimal and quiet: avoids noisy deprecation warnings and delegates to train_with_nas

import argparse
from src.train_with_nas import train as main_train


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--enable-nas", action="store_true")
    p.add_argument("--nas-layers")
    p.add_argument("--nas-weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args(argv)
    return main_train(
        model_name=args.model,
        epochs=args.epochs,
        enable_nas=args.enable_nas,
        nas_layers=args.nas_layers,
        nas_weight=args.nas_weight,
    )


if __name__ == '__main__':
    raise SystemExit(main())
