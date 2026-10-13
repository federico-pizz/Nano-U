"""Capture the regression oracle into ``baseline/``.

Not a test - pytest does not collect it. Run it once, on the code as it stands
*before* anything moves:

    conda run -n nano-u-gpu python -m tests.contract.capture_baseline

It copies the shipped artefacts, records their hashes, and re-scores the
held-out split with the current code. ``tests/contract/`` then compares the
repository against what this wrote. ``baseline/`` is gitignored: it is a local
oracle, not a deliverable.

The baseline is not a fresh low-epoch pipeline run, deliberately: the numbers
it pins are the ones the README quotes, which come from the *shipped* models,
and a fresh short run would both miss them and overwrite ``models/``. The shipped
artefacts are the stronger oracle, and capturing them cannot destroy anything.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime

from tests.contract import oracle


def git_revision(root):
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def capture_dataset(dataset, preset, out_dir, skip_metrics=False):
    """Write one dataset's slice of the baseline. Returns a summary dict."""
    paths = oracle.artifact_paths(dataset)
    os.makedirs(out_dir, exist_ok=True)

    missing = [k for k in ("tflite", "quant_params") if not os.path.exists(paths[k])]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: missing shipped artefact(s) {missing}. "
            f"Export them first:  python scripts/export.py --preset {preset}"
        )

    record = {"dataset": dataset, "preset": preset, "artifacts": {}}

    for key in ("tflite", "quant_params", "summary"):
        src = paths[key]
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
        record["artifacts"][key] = {
            "name": os.path.basename(src),
            "sha256": oracle.sha256_file(src),
            "size_bytes": os.path.getsize(src),
        }

    print(f"[{dataset}] copied {len(record['artifacts'])} artefacts")

    record["tflite_io"] = oracle.tflite_io(paths["tflite"])
    record["tflite_ops"] = oracle.tflite_op_codes(paths["tflite"])
    record["param_counts"] = {
        "float": oracle.float_param_count(),
        "qat": oracle.keras_param_count(paths["keras"]) if os.path.exists(paths["keras"]) else None,
        "int8": oracle.tflite_param_counts(paths["tflite"]),
    }
    print(
        f"[{dataset}] params float={record['param_counts']['float']} "
        f"qat={record['param_counts']['qat']} "
        f"int8={record['param_counts']['int8']['total']}"
    )

    if not skip_metrics:
        record["metrics"] = {}
        for tag, model_path in (("int8", paths["tflite"]), ("fp32", paths["keras"])):
            if not os.path.exists(model_path):
                continue
            metrics = oracle.score_split(preset, model_path, split="test")
            with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as fh:
                json.dump(metrics, fh, indent=2)
            record["metrics"][tag] = oracle.flat_metrics(metrics)
            print(
                f"[{dataset}] {tag}: "
                + "  ".join(f"{k}={v:.6f}" for k, v in record["metrics"][tag].items())
            )

    with open(os.path.join(out_dir, "record.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(oracle.DATASETS),
        help="dataset to capture (repeatable; default: all)",
    )
    parser.add_argument("--out", default="baseline", help="output directory (default: baseline)")
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="artefacts and parameter counts only, no scoring",
    )
    args = parser.parse_args(argv)

    root = oracle.project_root()
    out_root = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(out_root, exist_ok=True)

    datasets = args.dataset or sorted(oracle.DATASETS)
    records = {}
    for dataset in datasets:
        records[dataset] = capture_dataset(
            dataset,
            oracle.DATASETS[dataset],
            os.path.join(out_root, dataset),
            skip_metrics=args.skip_metrics,
        )

    manifest = {
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(root),
        "python": sys.version.split()[0],
        "datasets": {name: rec["preset"] for name, rec in records.items()},
        "note": "regression oracle; regenerate only on purpose",
    }
    with open(os.path.join(out_root, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nbaseline written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
