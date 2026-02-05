# API Reference

Current entry points after refactoring. See [REFACTURING_DOCUMENTATION.md](REFACTURING_DOCUMENTATION.md) for migration and usage details.

## Models (`src.models`)

```python
from src.models import (
    create_nano_u,
    create_bu_net,
    create_nano_u_functional,
    create_bu_net_functional,
    create_model_from_config,
    count_parameters,
    get_model_summary,
    get_model_config,
    validate_model_serialization,
)
```

- **create_nano_u**(input_shape=(48,64,3), filters=[16,32], bottleneck=64, name='nano_u') → Model  
- **create_bu_net**(input_shape=(48,64,3), filters=[32,64,128], bottleneck=256, name='bu_net') → Model  
- **create_model_from_config**(config) → Model — config must include `model_name`, optional `input_shape`, `filters`, `bottleneck`  
- **count_parameters**(model) → int  
- **get_model_summary**(model) → str  
- **get_model_config**(model) → dict  
- **validate_model_serialization**(model) → bool  

## Training (`src.train`)

```python
from src.train import train_model, train_single_model, train_with_distillation, train_step
```

- **train_model**(config_path='config/experiments.yaml', experiment_name='default', output_dir=None) → dict  
  Loads config, resolves experiment (top-level or `experiments.<name>`), runs training with synthetic data if no dataset paths. Returns `{status, final_metrics, model_path, experiment_dir}` or `{status: 'failed', error, traceback}`.

- **train_single_model**(model, config, train_data, val_data=None) → History  
- **train_with_distillation**(student, teacher, config, train_data, val_data=None) → History  
- **train_step**(student, teacher, x, y, optimizer, alpha=0.3, temperature=4.0) → dict  

CLI: `python src/train.py --config config/experiments.yaml --experiment quick_test --output results/`

## NAS (`src.nas`)

```python
from src.nas import compute_layer_redundancy, compute_nas_metrics, NASCallback, validate_nas_computation, analyze_model_redundancy
```

- **compute_layer_redundancy**(activations, eps=1e-6) → dict  
  Returns `redundancy_score`, `condition_number`, `rank`, `num_channels` (SVD-based, stable).

- **NASCallback**(layers_to_monitor=None, log_frequency=10, output_dir='nas_logs/', **kwargs)  
  Epoch-level callback: at end of each epoch writes redundancy metrics to `output_dir/metrics.csv`. Extra kwargs ignored.

- **validate_nas_computation**() → bool  
- **analyze_model_redundancy**(model, x, layers_to_monitor=None) → dict  

## Experiments (`src.experiment` and `scripts/run_experiments.py`)

- **run_experiment**(config_name, config_path='config/experiments.yaml', output_dir='results/') → dict  
- **run_experiment_sweep**(experiment_configs, config_path=..., output_dir=...) → list  

CLI:  
`python scripts/run_experiments.py --list`  
`python scripts/run_experiments.py --experiment quick_test --output results/`

## Benchmarks (`src.benchmarks`)

- **benchmark_inference**(model, input_shape=(48,64,3)) → dict  
- **validate_tflite_optimization**(model) → dict  
- **MemoryProfilingCallback**() — Keras callback for resource monitoring.

## Config

- **config/experiments.yaml** — Single file; experiments under `experiments:` (e.g. `quick_test`, `standard`, `distillation`).  
- **scripts/migrate_config.py** — Migrate old flat YAML to new format:  
  `python scripts/migrate_config.py old_config.yaml new_config.yaml`

## Config loading (`src.utils.config`)

```python
from src.utils.config import load_config
full = load_config("config/experiments.yaml")  # full dict; experiments in full["experiments"]
```

## Data (`src.data`)

```python
from src.data import make_dataset, get_synthetic_data
# make_dataset(img_files, mask_files, batch_size=8, ...) → tf.data.Dataset
```

---

Last updated: 2026-02-04
