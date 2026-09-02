"""Download a dataset and lay it out the way the pipeline reads it.

Writes `data/<name>/{train,val,test}/{img,mask}`, splitting each capture
sequence in time order - first 70% train, next 20% val, last 10% test - so no
two splits share a moment. Only frames that carry a ground-truth mask are kept.

Usage:
    python scripts/get_data.py                          # tinyagri
    python scripts/get_data.py --preset botanicgarden
    python scripts/get_data.py --dry-run                # show the plan, download nothing
    python scripts/get_data.py --verify                 # check what is already on disk
"""

import argparse
import sys

from nanou.config import resolve
from nanou.fetch import FetchError, download, plan, spec_for, verify


def main():
    parser = argparse.ArgumentParser(description="Download and lay out a dataset")
    parser.add_argument("--preset", default="tinyagri",
                        help="config preset from config/presets/ (default: tinyagri)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the split plan without downloading")
    parser.add_argument("--verify", action="store_true",
                        help="check the dataset already on disk and exit")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent downloads (default: 8)")
    args = parser.parse_args()

    resolved = resolve(args.preset)
    data_root = next(iter(resolved.models.values())).data_root

    try:
        if args.verify:
            counts = verify(data_root)
            print(f"{data_root}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
            return

        spec = spec_for(data_root)
        if not spec.repo_id:
            raise FetchError(spec.note or f"{spec.name} has no public release")

        if args.dry_run:
            work = plan(spec)
            total = sum(len(v) for v in work.values())
            print(f"{spec.name} -> {data_root}")
            for split, items in work.items():
                print(f"  {split}: {len(items)} frame pairs")
            print(f"  total: {total} pairs ({2 * total} files to download)")
            return

        print(f"Downloading {spec.name} from {spec.repo_id} into {data_root}")
        download(spec, data_root, workers=args.workers)
        counts = verify(data_root)
        print("\nDone: " + ", ".join(f"{k} {v} pairs" for k, v in counts.items()))

    except FetchError as e:
        print(f"\n{e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
