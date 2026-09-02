"""Dataset download and layout verification.

Turns a released dataset into the layout the pipeline reads:

    data/<name>/{train,val,test}/{img,mask}/<domain>_<scene>_frame<N>.png

The splits are **sequential within each capture sequence** - the first 70% of a
sequence's frames train, the next 20% validate, the last 10% test - so no two
splits share a moment in time. A random split would leak: consecutive frames of
a moving rover are near-duplicates.

Downloads go over plain HTTPS against the public dataset repository, using only
the standard library, so a fresh clone needs nothing beyond the pinned install.
"""

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .config import SPLITS

HF_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# Frame numbers are the only stable part of the source filenames: the image
# stems carry a per-scene capture prefix (d5_s1, d6_s1, ...) that the processed
# names drop.
IMAGE_FRAME = re.compile(r"frame(\d+)\.png$")
MASK_FRAME = re.compile(r"mask(\d+)\.png$")


class FetchError(RuntimeError):
    """A dataset could not be fetched or does not have a public release."""


@dataclass(frozen=True)
class DatasetSpec:
    """How to turn one released dataset into the processed layout.

    Attributes:
        name: directory name under `data/`.
        repo_id: dataset repository, or None when there is no public release.
        sequences: (domain, scene) pairs, each one capture sequence.
        image_dir / mask_dir: templates for the source directories.
        target_size: (width, height) the frames are stored at.
        fractions: (train, val) fractions; test takes the remainder.
        note: shown when there is no public release.
    """

    name: str
    repo_id: Optional[str] = None
    sequences: Sequence[Tuple[str, str]] = ()
    image_dir: str = "{domain}/{scene}"
    mask_dir: str = "SAM2-masks/{domain}/{scene}"
    target_size: Tuple[int, int] = (80, 60)
    fractions: Tuple[float, float] = (0.70, 0.20)
    note: str = ""


TINYAGRI = DatasetSpec(
    name="TinyAgri",
    repo_id="federico-pizz/TinyAgri",
    sequences=[(d, s) for d in ("Crops", "Tomatoes") for s in ("scene1", "scene2")],
)

# Botanic Garden is a third-party benchmark; only its derived masks are ours, so
# there is nothing here to download on the user's behalf.
BOTANICGARDEN = DatasetSpec(
    name="BotanicGarden",
    note=(
        "Botanic Garden has no packaged release. Get the raw sequences from "
        "https://github.com/robot-pesg/BotanicGarden, then lay them out as "
        "data/BotanicGarden/{train,val,test}/{img,mask} split by contiguous "
        "sequence."
    ),
)

DATASETS: Dict[str, DatasetSpec] = {d.name: d for d in (TINYAGRI, BOTANICGARDEN)}


def spec_for(data_root) -> DatasetSpec:
    """Look up the dataset spec for a preset's `data_root`.

    Raises:
        FetchError: the directory name matches no known dataset.
    """
    name = Path(data_root).name
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise FetchError(f"no download is defined for '{name}' (known: {known})") from None


# -- source listing -------------------------------------------------------------

def _get(url: str, retries: int = 3) -> bytes:
    last: Optional[Exception] = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:  # transient
            last = exc
    raise FetchError(f"could not fetch {url}: {last}")


def _list_dir(repo_id: str, path: str) -> List[str]:
    """Return the file paths in one directory of the dataset repository."""
    entries = json.loads(_get(HF_TREE.format(repo=repo_id, path=path)))
    return [e["path"] for e in entries if e.get("type") == "file"]


def _by_frame(paths: Sequence[str], pattern: re.Pattern) -> Dict[int, str]:
    found = {}
    for path in paths:
        match = pattern.search(path)
        if match:
            found[int(match.group(1))] = path
    return found


def sequential_split(items: Sequence, train: float, val: float) -> Tuple[List, List, List]:
    """Split a sequence in capture order into train / val / test.

    Rounding is to the nearest frame, and test takes whatever is left, so the
    three parts always add back up to the input.
    """
    n = len(items)
    n_train = round(train * n)
    n_val = round(val * n)
    return (list(items[:n_train]),
            list(items[n_train:n_train + n_val]),
            list(items[n_train + n_val:]))


def plan(spec: DatasetSpec) -> Dict[str, List[Tuple[str, str, str]]]:
    """Work out what to download, without downloading anything.

    Returns:
        Split name -> list of (output filename, image source path, mask source
        path). A frame is included only when it has both an image and a mask;
        the mask set is the smaller one, since only part of each sequence is
        annotated.

    Raises:
        FetchError: the dataset has no public release.
    """
    if not spec.repo_id:
        raise FetchError(spec.note or f"{spec.name} has no public release")

    out: Dict[str, List[Tuple[str, str, str]]] = {s: [] for s in SPLITS}
    for domain, scene in spec.sequences:
        images = _by_frame(
            _list_dir(spec.repo_id, spec.image_dir.format(domain=domain, scene=scene)),
            IMAGE_FRAME)
        masks = _by_frame(
            _list_dir(spec.repo_id, spec.mask_dir.format(domain=domain, scene=scene)),
            MASK_FRAME)
        frames = sorted(set(images) & set(masks))
        parts = sequential_split(frames, *spec.fractions)
        for split, numbers in zip(SPLITS, parts):
            out[split].extend(
                (f"{domain}_{scene}_frame{n}.png", images[n], masks[n]) for n in numbers)
    return out


# -- download and write ---------------------------------------------------------

def _fetch_resized(repo_id: str, path: str, size: Tuple[int, int],
                   mode: str, resample: int) -> Image.Image:
    raw = _get(HF_FILE.format(repo=repo_id, path=path))
    return Image.open(BytesIO(raw)).convert(mode).resize(size, resample)


def _write_pair(spec: DatasetSpec, out_root: Path, split: str,
                item: Tuple[str, str, str]) -> None:
    name, image_path, mask_path = item
    # Bilinear for the image and nearest for the mask: the mask is a label map,
    # so an interpolated edge pixel would be a class that does not exist.
    image = _fetch_resized(spec.repo_id, image_path, spec.target_size,
                           "RGB", Image.Resampling.BILINEAR)
    mask = _fetch_resized(spec.repo_id, mask_path, spec.target_size,
                          "L", Image.Resampling.NEAREST)
    image.save(out_root / split / "img" / name)
    mask.save(out_root / split / "mask" / name)


def download(spec: DatasetSpec, out_root, workers: int = 8,
             progress=print) -> Dict[str, int]:
    """Download a dataset and write the processed splits under *out_root*.

    Args:
        spec: which dataset, from `DATASETS`.
        out_root: destination, normally `data/<name>`.
        workers: concurrent downloads.
        progress: called with one status line per sequence.

    Returns:
        Split name -> number of frame pairs written.
    """
    out_root = Path(out_root)
    work = plan(spec)
    for split in SPLITS:
        (out_root / split / "img").mkdir(parents=True, exist_ok=True)
        (out_root / split / "mask").mkdir(parents=True, exist_ok=True)

    written = {}
    for split, items in work.items():
        progress(f"  {split}: {len(items)} frame pairs")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: _write_pair(spec, out_root, split, item), items))
        written[split] = len(items)
    return written


# -- verification ---------------------------------------------------------------

def verify(out_root) -> Dict[str, int]:
    """Check a processed dataset and return its per-split pair count.

    Raises:
        FetchError: a split is missing, empty, or has unpaired images and masks.
    """
    out_root = Path(out_root)
    counts = {}
    for split in SPLITS:
        img_dir, mask_dir = out_root / split / "img", out_root / split / "mask"
        if not (img_dir.is_dir() and mask_dir.is_dir()):
            raise FetchError(f"{out_root}: missing {split}/img or {split}/mask")
        images = {p.name for p in img_dir.glob("*.png")}
        masks = {p.name for p in mask_dir.glob("*.png")}
        if not images:
            raise FetchError(f"{out_root}: {split}/img holds no PNGs")
        if images != masks:
            missing = sorted(images ^ masks)[:5]
            raise FetchError(
                f"{out_root}: {split} images and masks do not pair up "
                f"({len(images)} vs {len(masks)}); first unmatched: {missing}")
        counts[split] = len(images)
    return counts


__all__ = [
    "DatasetSpec",
    "DATASETS",
    "FetchError",
    "TINYAGRI",
    "BOTANICGARDEN",
    "spec_for",
    "sequential_split",
    "plan",
    "download",
    "verify",
]
