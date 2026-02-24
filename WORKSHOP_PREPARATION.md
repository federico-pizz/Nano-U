# Nano-U: Pre-Conference Workshop Preparation List

This checklist outlines the essential steps and verifications required before presenting the Python/TensorFlow side of the **Nano-U** project at a conference workshop. No new features are added; this focuses purely on stability, reproducibility, and instructional readiness.

---

## 1. Environment & Dependencies

Before the workshop begins, ensure the runtime environment is robust and reproducible for attendees.

- [ ] **Verify `requirements.txt`:** Ensure all pinned versions (e.g., `tensorflow==2.15.0`, `opencv-python`) install cleanly without conflicts on both Linux and Windows (WSL) setups.
- [ ] **Test Virtual Environment Creation:** Run a fresh `python -m venv .venv-tf && source .venv-tf/bin/activate && pip install -r requirements.txt` from scratch to catch any missing sub-dependencies.
- [ ] **Check Hardware Acceleration:** If presenting on a GPU-enabled machine, run a miniature script to verify TensorFlow detects the CUDA/cuDNN stack (`tf.config.list_physical_devices('GPU')`). If targeting CPU-only for attendees, ensure the pipeline runs without throwing fatal XLA errors.

## 2. Dataset Readiness

The workshop will likely rely on the `data/processed_data/` directory.

- [ ] **Validate Dataset Structure:** Ensure the `train/`, `val/`, and `test/` folders exist and contain aligned image/mask pairs.
- [ ] **Check Pathing in `config.yaml`:** Verify that `config.yaml` uses relative paths correctly so that attendees cloning the repo don't encounter absolute path `FileNotFoundError`s.
- [ ] **Verify `make_dataset` Loading:** Run `src/data.py` directly or via a fast test to ensure `cv2.imread` correctly parses the images and Normalization bounds `[-1, 1]` are strictly applied.

## 3. Configuration & Experiments

The `config/experiments.yaml` dictates the flow of the workshop.

- [ ] **Test the `quick_test` Configuration:** Execute `python scripts/train_standard.py --experiment quick_test`. Verify it completes its 2 epochs within seconds/minutes and successfully triggers the automatic Post-Training Quantization script at the end.
- [ ] **Review NAS Hyperparameters:** In `experiments.yaml`, ensure the `nas_search` configuration has a reasonable `population_size` (e.g., 4) and `generations` (e.g., 3) so that a live demonstration of Evolutionary Search finishes within a workshop-friendly timeframe (under 10 minutes).
- [ ] **Check Distillation Parameters:** Ensure the `alpha` (loss weighting) and `temperature` values in the config are set to their proven defaults so the Student visibly learns from the Teacher during the demo.

## 4. Pipeline Execution & Output Verification

Run every major pipeline script end-to-end to verify the artifacts generated.

- [ ] **Standard Training:** Run `scripts/train_standard.py`. Verify it produces `results.json` and `best_model.keras` in the `results/` folder.
- [ ] **Distillation Training:** Run `scripts/train_distillation.py`. Verify both the Teacher and the Student are trained and saved correctly.
- [ ] **NAS Execution:** Run `scripts/nas_search.py`. 
  - Verify it prints the SVD Redundancy metrics (Condition Number, Rank) to the console.
  - Verify it generates `best_arch.json`.
- [ ] **Quantization Step:** Verify `src/quantize_model.py` successfully reads the validation imagery as its Representative Dataset and outputs an INT8 `.tflite` model (check file size: should be under ~200KB for Nano-U).
- [ ] **Evaluation:** Run `src/evaluate.py` to ensure it successfully generates the Matplotlib visual comparisons (`predictions.png`) highlighting Ground Truth vs Predicted masks.

## 5. Documentation & Interactive Materials

Presentations often require participants to follow along. The documentation must be bulletproof.

- [ ] **Walkthrough the `Nano_U_Colab_Guide.ipynb`:** Run the entire Colab notebook top-to-bottom precisely as an attendee would. Guarantee no cells throw errors.
- [ ] **Verify `README.md` Instructions:** Follow your own Quick Start guide literally. Copy-paste the commands into a fresh terminal to ensure typos don't derail the presentation.
- [ ] **Review `ARCHITECTURE_AND_FEATURES.md`:** Skim the citations and function mappings to ensure you can confidently answer questions about the MobileNet/AmoebaNet/SVD origins.

## 6. Cleanup

- [ ] Clear out heavy, leftover artifact folders (`logs/`, `nas_logs/`, old `results/`) before zipping or pushing the final presentation-ready commit to GitHub.
