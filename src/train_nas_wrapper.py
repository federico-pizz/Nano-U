# Deprecated shim: prefer src/train_with_nas.py
import warnings

warnings.warn("train_nas_wrapper.py is deprecated; call src/train_with_nas.py directly", DeprecationWarning)

try:
    from src.train_with_nas import train as main_train
except Exception:
    main_train = None


def main(argv=None):
    if main_train is None:
        raise ImportError("src.train_with_nas not available; please use src/train.py or src/train_with_nas.py directly")
    # simple CLI passthrough for compatibility
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--enable-nas", action="store_true")
    p.add_argument("--nas-layers")
    p.add_argument("--nas-weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args(argv)
    return main_train(model_name=args.model, epochs=args.epochs, enable_nas=args.enable_nas, nas_layers=args.nas_layers, nas_weight=args.nas_weight)


if __name__ == '__main__':
    raise SystemExit(main())
