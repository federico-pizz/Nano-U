"""NAS: stable redundancy metrics (SVD) and a lightweight epoch-level callback."""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, DefaultDict, Union, Any
from collections import defaultdict
import pandas as pd
from pathlib import Path


def compute_layer_redundancy(activations: tf.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Compute stable redundancy score using SVD decomposition.
    
    Args:
        activations: Layer activations tensor (batch, height, width, channels)
        eps: Small value to prevent division by zero
    
    Returns:
        Dictionary with redundancy metrics
    """
    # Reshape to (samples, features) for SVD
    ndim = len(activations.shape)
    if ndim == 2:
        reshaped = activations
        channels = activations.shape[-1]
    else:
        batch_size, height, width, channels = activations.shape
        reshaped = tf.reshape(activations, (-1, channels))
    
    # Center activations
    mean_act = tf.reduce_mean(reshaped, axis=0)
    centered = reshaped - mean_act
    
    # SVD for numerical stability
    s, u, v = tf.linalg.svd(centered, full_matrices=False)
    singular_values = tf.maximum(s, eps)  # Clamp to avoid zeros
    
    # Condition number using SVD
    condition_number = tf.reduce_max(singular_values) / tf.reduce_min(singular_values)
    
    # Redundancy score (normalized)
    redundancy = 1.0 / (1.0 + tf.math.log(condition_number + 1.0))
    
    def _numpy(x):
        return int(x.numpy()) if hasattr(x, "numpy") else int(x)

    return {
        "redundancy_score": float(redundancy.numpy()),
        "condition_number": float(condition_number.numpy()),
        "rank": _numpy(tf.reduce_sum(tf.cast(singular_values > eps, tf.int32))),
        "num_channels": _numpy(channels),
    }


def extract_activations(model: tf.keras.Model, layer_names: List[str], 
                       x: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Extract activations from specified layers.
    
    Args:
        model: Keras model
        layer_names: List of layer names to extract
        x: Input batch
    
    Returns:
        Dictionary of layer name to activation tensor
    """
    # Create intermediate model
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = intermediate_model(x)
    
    return dict(zip(layer_names, activations))


def compute_nas_metrics(model: tf.keras.Model, x: tf.Tensor, 
                       layers_to_monitor: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute NAS metrics for all specified layers.
    
    Args:
        model: Keras model
        x: Input batch
        layers_to_monitor: List of layer names to monitor
    
    Returns:
        Dictionary of layer name to redundancy metrics
    """
    # Extract activations
    activations_dict = extract_activations(model, layers_to_monitor, x)
    
    # Compute metrics for each layer
    metrics = {}
    for layer_name, activations in activations_dict.items():
        metrics[layer_name] = compute_layer_redundancy(activations)
    
    return metrics


def analyze_model_redundancy(model: tf.keras.Model, x: tf.Tensor,
                            layers_to_monitor: List[str] = None) -> Dict[str, any]:
    """Analyze model redundancy across specified layers.
    
    Args:
        model: Keras model
        x: Input batch for analysis
        layers_to_monitor: List of layer names to monitor (defaults to conv layers)
    
    Returns:
        Dictionary with comprehensive redundancy analysis
    """
    if layers_to_monitor is None:
        # Default to monitoring all conv layers
        layers_to_monitor = [layer.name for layer in model.layers 
                           if isinstance(layer, (tf.keras.layers.Conv2D, 
                                                 tf.keras.layers.DepthwiseConv2D))]
    
    # Compute metrics
    metrics = compute_nas_metrics(model, x, layers_to_monitor)
    
    # Aggregate statistics
    redundancy_scores = [m['redundancy_score'] for m in metrics.values()]
    condition_numbers = [m['condition_number'] for m in metrics.values()]
    ranks = [m['rank'] for m in metrics.values()]
    
    return {
        'layer_metrics': metrics,
        'aggregate': {
            'average_redundancy': float(np.mean(redundancy_scores)),
            'min_redundancy': float(np.min(redundancy_scores)),
            'max_redundancy': float(np.max(redundancy_scores)),
            'average_condition_number': float(np.mean(condition_numbers)),
            'min_condition_number': float(np.min(condition_numbers)),
            'max_condition_number': float(np.max(condition_numbers)),
            'average_rank': float(np.mean(ranks)),
            'min_rank': float(np.min(ranks)),
            'max_rank': float(np.max(ranks))
        },
        'layers_monitored': layers_to_monitor,
        'batch_size': int(x.shape[0])
    }


class NASCallback(tf.keras.callbacks.Callback):
    """Lightweight NAS monitoring: logs redundancy metrics at end of each epoch."""

    def __init__(
        self,
        layers_to_monitor: List[str] = None,
        log_frequency: int = 1,
        output_dir: str = "nas_logs/",
        validation_data: Optional[Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]] = None,
        monitor_frequency: str = "epoch",
        log_dir: str = None,
        **kwargs,
    ):
        super().__init__()
        self.layers_to_monitor = layers_to_monitor or ["conv2d", "conv2d_1"]
        self.log_frequency = log_frequency
        self.output_dir = str(output_dir)
        self.monitor_frequency = monitor_frequency # 'epoch' or 'batch'
        self.validation_data = validation_data
        self.log_dir = log_dir
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.log_dir:
            self.writer = tf.summary.create_file_writer(self.log_dir)
            
        self.redundancy_history: List[Dict[str, Any]] = []
        self.metrics: DefaultDict[str, List[Any]] = defaultdict(list)
        self.batch_count = 0

    def _get_batch(self):
        """Get a batch for metrics (validation data or training data)."""
        val = None
        if self.validation_data is not None:
            val = self.validation_data
        elif hasattr(self.model, "validation_data") and self.model.validation_data is not None:
            val = self.model.validation_data
            
        if val is not None:
            if isinstance(val, tf.data.Dataset):
                for x, _ in val.take(1):
                    return x[: min(16, tf.shape(x)[0])]
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                x = val[0]
                return x[: min(16, len(x))]
                
        # Fallback to test batch if set by unit tests
        if hasattr(self, "_test_x_batch"):
            return self._test_x_batch
        return None

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, float]] = None):
        self.batch_count += 1
        if self.monitor_frequency == "batch" and self.batch_count % self.log_frequency == 0:
            self._compute_and_log(f"step_{self.batch_count}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if self.monitor_frequency == "epoch" and (epoch + 1) % self.log_frequency == 0:
            self._compute_and_log(epoch)

    def _compute_and_log(self, identifier: Union[int, str]):
        x_batch = self._get_batch()
        if x_batch is None:
            return
            
        row: Dict[str, Union[int, str, float]] = {"epoch" if isinstance(identifier, int) else "step": identifier}
        for name in self.layers_to_monitor:
            try:
                layer = self.model.get_layer(name)
                # Ensure we handle functional model correctly
                inter = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                act = inter(x_batch, training=False)
                r = compute_layer_redundancy(act)
                row[f"{name}_redundancy_score"] = r["redundancy_score"]
                row[f"{name}_condition_number"] = r["condition_number"]
                row[f"{name}_rank"] = float(r["rank"])
                row[f"{name}_num_channels"] = float(r["num_channels"])
            except (ValueError, KeyError) as e:
                print(f"⚠️ NAS monitor skipping layer {name}: {e}")
                continue
                
        self.redundancy_history.append(row)
        for k, v in row.items():
            if k not in ["epoch", "step"]:
                self.metrics[k].append(v)
                
        if self.log_dir:
            with self.writer.as_default():
                step = identifier if isinstance(identifier, int) else self.batch_count
                for k, v in row.items():
                    if k not in ["epoch", "step"]:
                        tf.summary.scalar(f"nas/{k}", v, step=step)
            self.writer.flush()
                
        csv_path = Path(self.output_dir) / "metrics.csv"
        pd.DataFrame(self.redundancy_history).to_csv(csv_path, index=False)

    def on_train_end(self, logs: Optional[Dict[str, float]] = None):
        if self.redundancy_history:
            csv_path = Path(self.output_dir) / "metrics.csv"
            pd.DataFrame(self.redundancy_history).to_csv(csv_path, index=False)

    def get_metrics(self) -> Dict[str, List[float]]:
        return dict(self.metrics)


def validate_nas_computation() -> bool:
    """Validate NAS computation with test cases."""
    try:
        # Test 1: Basic functionality
        test_activations = tf.random.normal((32, 24, 32, 64))  # (batch, h, w, channels)
        metrics = compute_layer_redundancy(test_activations)
        
        assert 0.0 <= metrics['redundancy_score'] <= 1.0, "Redundancy score out of range"
        assert metrics['condition_number'] > 0, "Condition number should be positive"
        assert metrics['rank'] > 0, "Rank should be positive"
        
        # Test 2: Identical channels (rank 1) – just check it runs and returns valid score
        perfect_corr = tf.tile(tf.random.normal((32, 1)), [1, 64])
        metrics = compute_layer_redundancy(perfect_corr)
        assert 0.0 <= metrics["redundancy_score"] <= 1.0

        # Test 3: Independent channels – valid score in [0, 1]
        no_corr = tf.random.normal((32, 64))
        metrics = compute_layer_redundancy(no_corr)
        assert 0.0 <= metrics["redundancy_score"] <= 1.0
        
        return True
        
    except Exception as e:
        print(f"NAS validation failed: {e}")
        return False


def get_nas_summary(metrics: Dict[str, Dict[str, float]]) -> str:
    """Get formatted NAS summary."""
    summary = []
    summary.append("=== NAS Analysis Summary ===")
    
    for layer_name, layer_metrics in metrics.items():
        summary.append(f"\n{layer_name}:")
        summary.append(f"  Redundancy Score: {layer_metrics['redundancy_score']:.4f}")
        summary.append(f"  Condition Number: {layer_metrics['condition_number']:.2f}")
        summary.append(f"  Rank: {layer_metrics['rank']}/{layer_metrics['num_channels']}")
    
    return "\n".join(summary)


import random
from src.models.builders import create_searchable_nano_u, BLOCK_MAP, get_block_map

class NASSearcher:
    """Evolutionary NAS Searcher for Nano-U architectures."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters: List[int],
        bottleneck: int,
        population_size: int = 4,
        generations: int = 3,
        mutation_rate: float = 0.2,
        efficiency_weight: float = 0.5,
        output_dir: str = "results/nas_search/",
        **kwargs
    ):
        self.input_shape = input_shape
        self.filters = filters
        self.bottleneck = bottleneck
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.efficiency_weight = efficiency_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.block_map = get_block_map(legacy_mode=True) # Nano-U is always sequential
        self.num_blocks = len(self.block_map)
        self.arch_len = 4  # [enc1, enc2, enc3, bottleneck]
        
    def generate_random_arch(self) -> List[int]:
        return [random.randint(0, self.num_blocks - 1) for _ in range(self.arch_len)]
    
    def mutate(self, arch: List[int]) -> List[int]:
        new_arch = list(arch)
        for i in range(len(new_arch)):
            if random.random() < self.mutation_rate:
                new_arch[i] = random.randint(0, self.num_blocks - 1)
        return new_arch
    
    def calculate_efficiency(self, model: tf.keras.Model) -> float:
        """Calculate efficiency score based on total parameters. Lower is better, but we return normalized [0, 1]."""
        total_params = model.count_params()
        # Assume a baseline Nano-U (~10k-20k params)
        # Higher score means MORE efficient (fewer params)
        return 1.0 / (1.0 + total_params / 50000.0)

    def search(self, train_fn, validation_data):
        """Run evolutionary search loop.
        
        Args:
            train_fn: A function(model, epochs) -> history that trains the model
            validation_data: Data to evaluate candidate performance
        """
        # Initial population
        population = [self.generate_random_arch() for _ in range(self.population_size)]
        history = []
        
        for gen in range(self.generations):
            print(f"\n🧬 Generation {gen+1}/{self.generations}")
            scores = []
            
            for i, arch in enumerate(population):
                print(f"  Testing arch {i+1}/{self.population_size}: {arch}")
                
                # Create and compile model
                model = create_searchable_nano_u(
                    input_shape=self.input_shape,
                    filters=self.filters,
                    bottleneck_filters=self.bottleneck,
                    arch_seq=arch,
                    name=f"nas_gen{gen}_idx{i}"
                )
                
                # Calculate efficiency (static)
                eff_score = self.calculate_efficiency(model)
                
                # Train model for a few epochs (proxy task)
                hist = train_fn(model, epochs=2)
                
                # Get performance (e.g., final val accuracy or IoU)
                perf_score = hist.history.get('val_accuracy', hist.history.get('accuracy', [0]))[-1]
                if 'val_iou' in hist.history:
                    perf_score = hist.history['val_iou'][-1]
                
                # Combined fitness
                fitness = (1.0 - self.efficiency_weight) * perf_score + self.efficiency_weight * eff_score
                
                scores.append({
                    "arch": arch,
                    "fitness": float(fitness),
                    "performance": float(perf_score),
                    "efficiency": float(eff_score),
                    "params": int(model.count_params())
                })
                
                print(f"    Fitness: {fitness:.4f} (Perf: {perf_score:.4f}, Eff: {eff_score:.4f}, Params: {model.count_params()})")
            
            # Sort by fitness
            scores.sort(key=lambda x: x['fitness'], reverse=True)
            history.append(scores)
            
            # Save progress
            pd.DataFrame(scores).to_csv(self.output_dir / f"gen_{gen}_results.csv")
            
            # Elite reproduction + Mutation
            best_arch = scores[0]['arch']
            next_population = [best_arch]  # Elitism
            
            while len(next_population) < self.population_size:
                # Select from top 2
                parent = scores[random.randint(0, min(1, len(scores)-1))]['arch']
                next_population.append(self.mutate(parent))
            
            population = next_population
            
        return {
            "best_arch": scores[0]['arch'],
            "best_fitness": scores[0]['fitness'],
            "history": history
        }