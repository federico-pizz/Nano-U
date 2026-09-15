# Nano-U - Paper vs repository

The README states what the code does now. This file is for a reader arriving
from the paper: it reconciles the numbers published in the
[TAROS 2026 paper](../README.md#citation) (arXiv:2605.10210) with the ones this
repository produces today, and says what accounts for each gap.

Both sets are reproducible, and the difference is which config you run. The
`-paper` presets carry the submission's hyperparameters; the `v1.2.0` tag is
the code as it stood at submission.

| | Paper (arXiv:2605.10210) | This repo (current) |
|:---|:---|:---|
| Preset | `botanicgarden-paper`, `tinyagri-paper` | `botanicgarden`, `tinyagri` |
| Distillation ($T$, $\alpha$) | BG 8.0 / 0.3 | TinyAgri 4.0 / 0.5 | BG 4.0 / 0.3 | TinyAgri 0.7 / 0.3 |
| Supervised loss | BCE only | BG: BCE | TinyAgri: 0.3*BCE + 0.7*Tversky (0.8/0.2) |
| Botanic Garden mIoU (FP32 / INT8) | 0.870 / 0.870 | 0.876 / 0.876 |
| TinyAgri mIoU (FP32 / INT8) | 0.883 / **0.700** | 0.884 / 0.884 |
| Inference latency | 845 ms (1.2 FPS) | 581 ms (~1.7 FPS), 426 ms on `multicore` |
| Peak Data RAM | 257 KB | 281 KB |
| Power | 470 mW board (137 mW idle, 333 mW active) | unchanged |

Three changes account for the gaps:

- **The TinyAgri INT8 collapse (0.883 -> 0.700) is attributable to a calibration plumbing bug rather than to QAD itself.** `export_int8` did not forward `--config` to the quantizer, so INT8 calibration always read whatever the single template config happened to contain, regardless of which config trained the model - calibrating on another domain's images, or on random noise when the paths were placeholders. Botanic Garden was unaffected because it was the domain that template was filled in for. The config is now threaded through, and missing calibration images raise instead of silently degrading to noise.
- **The tuned configs** come from the leakage-safe grouped CV sweep (`scripts/sweep.py`) run after submission; TinyAgri in particular moved to a precision-favoring Tversky blend and a sharpening $T<1$. The sweep files for each stage are in `sweeps/`, and what each stage picked is in [`DECISIONS.md`](DECISIONS.md).
- **Latency and RAM** reflect the buffer-reuse work in the inference engine: faster (845 -> 581 ms) at the cost of a slightly larger static arena (257 -> 281 KB).

What this table cannot say yet is how much of the 0.700 -> 0.884 move is the
bug fix alone and how much is the tuning, or whether 0.870 vs 0.876 is outside
run-to-run noise. Those are questions A2 and A4 in
[`PAPER_QUESTIONS.md`](PAPER_QUESTIONS.md); `experiments/seed_variance.py`
gives the error bars.
