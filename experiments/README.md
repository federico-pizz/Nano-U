# Experiments

Claims that a unit test cannot settle, because settling them takes a training
run. `tests/` checks that the code behaves correctly; this folder measures what
the model actually does.

Each file answers one question and produces one table, written to
`results/experiments/<name>/summary.json`.
That file is tracked by git; the runs under `runs/` (weights, figures, per-run
reports) stay local. The experiments use `nanou` like any other caller: they
add no library code, and nothing in `scripts/` or `src/` depends on them.

Every experiment trains the preset's student and distils from the preset's
shipped teacher (`models/<dataset>/bu_net.h5`); none of them retrains the
teacher. Its weights go under the run directory, never over the shipped
model. `--epochs 1` checks that an experiment runs end to end in a few minutes,
but the numbers it produces mean nothing.

| File | Question | Runs | Full runtime |
|:---|:---|:---:|:---|
| `ablation_qad.py` | Is QAD better than QAT-only, KD-only or plain PTQ? | 4 per seed (`--seeds`) | about 5 h per seed |
| `ablation_calibration.py` | Which variants need real calibration images to survive INT8? | 0 (re-exports) | minutes |
| `ablation_temperature.py` | Does the T**2 factor matter at the tuned T = 0.7, and how sensitive is the student to T? | 9 | about 11 h |
| `seed_variance.py` | Are the headline numbers outside run-to-run noise? | 5 (`--seeds`) | about 6 h |

The runtimes are estimates for TinyAgri on a laptop RTX 5060: an epoch takes
about 25 s, the preset allows 200 with early stopping, and the shipped student
stopped at 176. A run is then about 75 minutes, INT8 export and both evaluations
included. The smoke runs (`--epochs 1`) took about four minutes for
`ablation_qad.py`. `ablation_calibration.py` trains nothing, so its cost is two
exports and two test evaluations per model, a few minutes for a whole ablation.

```bash
python experiments/ablation_qad.py --preset tinyagri --seeds 3
python experiments/ablation_calibration.py --preset tinyagri   # after the one above
python experiments/ablation_temperature.py --preset tinyagri
python experiments/seed_variance.py --preset tinyagri --seeds 5
```

## What each one claims

**`ablation_qad.py`.** The paper has no ablation. This runs the four cells of
teacher term {on, off} x QAT {on, off}. All four go through the distillation
step, so they share the preset's supervised loss (BCE plus its Tversky blend).
"Teacher off" is `alpha = 0`, and "QAT off" trains in float and leaves INT8 to
post-training quantization at export. The row to read is the INT8 mIoU and its
gap to FP32.

`--seeds N` trains every cell N times, from the preset's seed upward, and the
table then carries a per-cell mean and sample std. One run per cell is not
enough to rank them: the seed-to-seed spread is about 0.002 mIoU on TinyAgri and
0.023 on Botanic Garden, against differences between cells of about 0.01.

**`ablation_calibration.py`.** Trains nothing. It takes the models
`ablation_qad.py` left under its `runs/`, exports each one twice, once
calibrated on the val split and once on random noise, and scores both on test.
The QAD ablation exports everything with real images, which is the friendly
case; this is the one where QAT is supposed to pay off, since a QAT model
carries its own ranges and a float-trained one has to learn them from the
representative data. That also means the two exports of a QAT-trained variant
come out byte-identical: TFLite uses the fake-quant ranges and ignores the
representative dataset. Only the float-trained cells can move, which is the
point of the comparison.

**`ablation_temperature.py`.** The T**2 rescaling of the teacher term is
Hinton's correction for T > 1. At the tuned TinyAgri T = 0.7 it scales the
teacher term down by half instead. This runs T in {0.7, 1, 2, 4, 8} with the
factor on and off (`t_scale: squared | none`), with T = 1 run once. These are
test-split numbers, so the table is a sensitivity curve around the tuned T; the
T itself was chosen by the CV sweep, not by this table.

**`seed_variance.py`.** Retrains the student with N consecutive seeds, starting
at the preset's own seed so the first row is the shipped configuration, and
reports the mean and sample std of mIoU, Dice, precision, recall and F0.5 in
both formats. The teacher is fixed, so this is the student's variance, not the
whole pipeline's.

An encoder-transfer experiment is deferred: it needs both datasets present
at once, which an outside user cannot set up.
