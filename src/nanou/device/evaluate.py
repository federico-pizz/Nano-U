"""Run the model on the attached board and score what it sends back.

Scoring goes through `nanou.evaluate` - the same split loading, `score`, `plot`
and `write_report` the host evaluator uses - so host and device numbers are
comparable by construction rather than by coincidence. What is specific to the
device lives here: building the binary, reading its serial protocol, and
turning the hex rows back into an array.
"""

import os
import subprocess
from pathlib import Path

import numpy as np

from nanou.config import get_project_root, resolve
from nanou.device.flash import firmware_env
from nanou.device.protocol import decode_hex_row, parse_idx_us_line, parse_stats_line
from nanou.device.session import DeviceSession
from nanou.evaluate import (
    frame_groups,
    plot,
    print_headline,
    project_path,
    score,
    split_dataset,
    split_files,
    write_report,
)
from nanou.predictor import logits_to_probs

TIMEOUT_SECONDS = 300


def read_device_outputs(lines, width: int) -> dict:
    """Decode the inference binary's serial protocol.

    Args:
        lines: The board's output, one decoded line at a time - what
            `DeviceSession.lines()` yields.
        width: Image width. Each output row is `width` little-endian f32
            logits, hex encoded, so a row is ``width * 8`` characters.

    Returns:
        Dict with ``img_data`` (image index -> list of decoded rows),
        ``latency_us`` (index -> preprocess/inference microseconds) and
        ``stats`` (index -> the firmware's own min/max of the output).
    """
    img_data, latency_us, stats = {}, {}, {}
    current = None
    row_len = width * 8
    for line in lines:
        if line.startswith("BENCHMARK_START:"):
            print(f"\nDevice announced {int(line.split(':')[1])} images.")
        elif line.startswith("IMG_OUTPUT_START:"):
            current = int(line.split(":")[1])
            img_data[current] = []
        elif line.startswith("IMG_OUTPUT_END:"):
            current = None
        elif line.startswith("IMG_PREPROCESS:"):
            idx, us = parse_idx_us_line(line)
            latency_us.setdefault(idx, {})["preprocess"] = us
        elif line.startswith("IMG_INFERENCE:"):
            idx, us = parse_idx_us_line(line)
            latency_us.setdefault(idx, {})["inference"] = us
        elif line.startswith("IMG_STATS:"):
            idx, min_v, max_v = parse_stats_line(line)
            stats[idx] = {"min": min_v, "max": max_v}
        elif line == "BENCHMARK_DONE":
            break
        elif current is not None and len(line) == row_len:
            try:
                img_data[current].append(decode_hex_row(line))
            except ValueError:
                # A row garbled on the wire. The image ends up short a row and
                # device_probs drops it, rather than scoring a partial image.
                pass
    return {"img_data": img_data, "latency_us": latency_us, "stats": stats}


def run_inference_on_device(model_name="nanou", config_path="config/presets/tinyagri.yaml"):
    """Build and flash the inference binary, then read one run's output off the serial line.

    Args:
        model_name: Model whose baked-in weights the firmware is built with.
        config_path: Preset the firmware and the host must agree on.

    Returns:
        What `read_device_outputs` returns.
    """
    config = resolve(path=config_path)
    firmware = get_project_root() / "firmware"
    # The same quant-params refresh as scripts/flash.py: build.rs needs the JSON,
    # which is not committed, and a stale one would bake in another model's scales.
    env = firmware_env(model=model_name, config_path=config_path)

    print(f"[1/3] Building inference binary for {model_name}...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "inference"], cwd=firmware, check=True, env=env
    )

    print(f"\n[2/3] Flashing and running inference for {model_name}...")
    width = config.input_shape[1]
    with DeviceSession(
        ["cargo", "run", "--release", "--bin", "inference"],
        cwd=firmware,
        env=env,
        timeout=TIMEOUT_SECONDS,
    ) as session:
        return read_device_outputs(session.lines(), width)


def device_probs(img_data: dict, n_images: int, height: int) -> tuple[np.ndarray, list[int]]:
    """Stack the complete device outputs into probabilities, and say which images they are.

    An image whose rows did not all arrive is dropped rather than scored on
    partial data.

    Returns:
        ``(probs, indices)``: an (N, H, W, 1) float32 array and the split
        index of each of its rows.

    Raises:
        ValueError: if the board sent an index beyond the host's split, which
            means the firmware was built from a different image set.
        RuntimeError: if no image arrived complete.
    """
    kept, rows = [], []
    for idx in sorted(img_data):
        if idx >= n_images:
            raise ValueError(
                f"the device sent image {idx} but the test split has {n_images}: "
                "the firmware was built from a different image set"
            )
        if len(img_data[idx]) != height:
            print(
                f"Warning: image {idx} arrived with {len(img_data[idx])}/{height} rows, skipping."
            )
            continue
        kept.append(idx)
        rows.append(np.stack(img_data[idx], axis=0))
    if not kept:
        raise RuntimeError("the device returned no complete image")
    logits = np.stack(rows, axis=0)[..., np.newaxis].astype(np.float32)
    return logits_to_probs(logits), kept


def _latency_summary(latency_us: dict) -> dict:
    infer = [v["inference"] for v in latency_us.values() if "inference" in v]
    prep = [v["preprocess"] for v in latency_us.values() if "preprocess" in v]
    out = {}
    if infer:
        out["inference_mean_us"] = float(np.mean(infer))
        out["inference_min_us"] = int(min(infer))
        out["inference_max_us"] = int(max(infer))
    if prep:
        out["preprocess_mean_us"] = float(np.mean(prep))
    return out


def evaluate_esp_outputs(
    device_result,
    model_name="nanou",
    config_path="config/presets/tinyagri.yaml",
    threshold=0.5,
    samples_to_plot=6,
):
    """Score the board's outputs against the test split and write the figures and report.

    Row i of the device output is test image i in `split_files` order, which
    is the order build.rs embeds the images in. Outputs go to
    ``<results_dir>/<model_name>/`` as ``eval_device_test.png``,
    ``eval_device_test_metrics.png`` and ``eval_device_test.json``.

    Args:
        device_result: What `run_inference_on_device` returned.
        model_name: Names the output directory and the report.
        config_path: Preset that defines the test split and the normalization.
        threshold: Probability above which a pixel counts as terrain.
        samples_to_plot: How many frames end up in the qualitative figure.

    Returns:
        The metrics dict, with the device latency figures added.
    """
    config = resolve(path=config_path)
    imgs, masks = split_files(config, "test")
    inputs, gt = [], []
    for x, y in split_dataset(config, imgs, masks):
        inputs.append(x.numpy())
        gt.append(y.numpy())
    inputs, gt = np.concatenate(inputs, axis=0), np.concatenate(gt, axis=0)

    height = config.input_shape[0]
    probs, kept = device_probs(device_result["img_data"], len(gt), height)
    print(f"\nEvaluating {len(kept)} of {len(gt)} images for {model_name}...")
    groups = frame_groups(imgs, len(gt))
    results = score(probs, gt[kept], [groups[i] for i in kept] if groups else None, threshold)
    results.update(_latency_summary(device_result.get("latency_us", {})))
    print_headline(results, "Results from on-device inference")

    results_dir = config.output_dir or "results"
    out_dir = Path(project_path(os.path.join(results_dir, model_name)))
    norm = config.normalization
    plot(
        probs,
        gt[kept],
        inputs[kept],
        results,
        out_dir,
        "eval_device_test",
        norm["mean"],
        norm["std"],
        samples_to_plot,
    )
    write_report(
        results,
        out_dir / "eval_device_test.json",
        split="test",
        model=model_name,
        format="device",
        images_scored=len(kept),
    )
    return results
