# API Reference

Current entry points and module documentation for the Nano-U exploration framework.

## Pipeline (`src.pipeline`)

The primary entry point for automated research workflows.

- **run_training_pipeline**(config_name, config_path='config/experiments.yaml', output_dir='results/')  
  Runs a single training experiment with clean result management and logging.
- **run_pipeline_sweep**(experiment_configs, config_path='config/experiments.yaml', output_dir='results/sweeps/')  
  Executes multiple training pipelines sequentially with automatic failure handling.
- **run_end_to_end_pipeline**(config_path='config/experiments.yaml', output_base='results/')  
  Executes the full automated flow: Teacher training → Student distillation → Benchmarking.
- **run_nas_search**(config_path='config/experiments.yaml', output_dir='results/nas_search/')  
  Runs an evolutionary Architecture Search to find the optimal Nano-U configuration.

## Models (`src.models`)

Core architecture builders and utilities.

- **create_nano_u**(input_shape=(48,64,3), filters=[16,32], bottleneck=64, name='nano_u') → Model  
  Creates the ultra-lightweight student model.
- **create_bu_net**(input_shape=(48,64,3), filters=[32,64,128], bottleneck=256, name='bu_net') → Model  
  Creates the teacher model.
- **create_model_from_config**(config) → Model  
  Builds a model based on a configuration dictionary.
- **count_parameters**(model) → int  
  Returns total trainable parameter count.
- **get_model_summary**(model) → str  
  Returns a detailed string summary of the architecture.

## Training (`src.train`)

Low-level training primitives and custom loops.

- **train_model**(config_path='config/experiments.yaml', experiment_name='default', output_dir=None) → dict  
  High-level entry point resolving experiment settings and calling appropriate training functions.
- **train_single_model**(model, config, train_data, val_data=None) → History  
  Standard Keras training loop with integrated callbacks.
- **train_with_distillation**(student, teacher, config, train_data, val_data=None) → History  
  Custom training loop for knowledge distillation.
- **train_step**(student, teacher, x, y, optimizer, alpha=0.3, temperature=4.0) → dict  
  TensorFlow primitive for a single gradient update with distillation.

## Evaluation (`src.evaluate`)

Verification and visualization of model performance.

- **evaluate_and_plot**(model_name, config_path, batch_size=8, threshold=0.5, samples_to_plot=6, out_path=None)  
  Runs full test set evaluation, computes IOU/Dice/Focal metrics, and generates prediction visualizations.

## Benchmarks (`src.benchmarks`)

Inference performance and quantization validation.

- **benchmark_inference**(model, input_shape=(48,64,3)) → dict  
  Measures average latency (ms) and throughput (FPS).
- **validate_tflite_optimization**(model) → dict  
  Converts to TFLite and verifies quantized model size.
- **MemoryProfilingCallback**()  
  Keras callback for monitoring system and GPU memory usage.

## NAS (`src.nas`)

Neural Architecture Search and redundancy monitoring.

- **compute_layer_redundancy**(activations, eps=1e-6) → dict  
  SVD-based stability analysis for feature map redundancy.
- **NASCallback**(layers_to_monitor=None, log_frequency=10)  
  Callback for real-time redundancy monitoring during training.

## CLI Entry Points

- `python scripts/run_pipeline.py --config config/experiments.yaml --experiment [name]`
- `python src/evaluate.py --model-name nano_u --config config/experiments.yaml`
- `python scripts/migrate_config.py old_config.yaml new_config.yaml`
