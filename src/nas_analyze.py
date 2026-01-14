"""
Minimal analysis runner to compute layer redundancy scores using
DistillationAwareNAS or RunningCovariance. Intended to be invoked by the
scripts/tf_pipeline.py analyze subcommand.

Usage examples:
  python -m src.nas_analyze --teacher model_teacher.keras --mapping mapping.json --out results/nas/scores.json
"""
from pathlib import Path
import argparse
import json
import tensorflow as tf
from src.nas_covariance import DistillationAwareNAS


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nas_analyze")
    parser.add_argument("--teacher", required=True, help="Path to teacher model (keras)")
    parser.add_argument("--mapping", required=True, help="JSON file mapping teacher->student layers or list of teacher layers to analyze")
    parser.add_argument("--out", required=True, help="Output JSON path for scores")
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args(argv)

    teacher = load_model(args.teacher)

    # mapping may be a simple list of layer names (analyze those layers only)
    mapping_data = json.loads(Path(args.mapping).read_text())
    if isinstance(mapping_data, dict):
        # require a student model if dict mapping provided
        raise SystemExit("mapping as dict not supported by minimal runner; provide a list of teacher layers")

    # Build a lightweight RunningCovariance-based analyzer per layer
    from src.nas_covariance import RunningCovariance, ActivationExtractor

    extractor = ActivationExtractor(teacher, mapping_data)
    layer_redundancies = {name: RunningCovariance(int(teacher.get_layer(name).output.shape[-1])) for name in mapping_data}

    # Create a tiny synthetic dataset for analysis if no real data available
    ds = tf.data.Dataset.from_tensor_slices((tf.random.normal((args.num_batches * 4, 256, 256, 3)), tf.zeros((args.num_batches * 4, 1))))
    ds = ds.batch(4)

    for i, (x, _) in enumerate(ds.take(args.num_batches)):
        acts = extractor(x, training=False)
        for name, act in acts.items():
            layer_redundancies[name].update(act)

    scores = {name: float(rc.redundancy_score().numpy()) for name, rc in layer_redundancies.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2))
    print("Wrote scores to", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
