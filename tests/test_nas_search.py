import os
import sys
import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nas import NASSearcher
from src.models.builders import create_searchable_nano_u

def test_nas_searcher_initialization():
    searcher = NASSearcher(
        input_shape=(48, 64, 3),
        filters=[16, 32, 64],
        bottleneck=64,
        population_size=2,
        generations=1
    )
    assert searcher.population_size == 2
    assert searcher.generations == 1
    assert searcher.arch_len == 4

def test_generate_random_arch():
    searcher = NASSearcher((48, 64, 3), [16, 32, 64], 64)
    arch = searcher.generate_random_arch()
    assert len(arch) == 4
    assert all(0 <= x < searcher.num_blocks for x in arch)

def test_mutate():
    searcher = NASSearcher((48, 64, 3), [16, 32, 64], 64, mutation_rate=1.0)
    arch = [0, 0, 0, 0]
    mutated = searcher.mutate(arch)
    assert len(mutated) == 4
    # With mutation_rate=1.0, it's very likely at least one changes, 
    # but since it's random it could pick 0 again. 
    # We just check it's a valid arch.
    assert all(0 <= x < searcher.num_blocks for x in mutated)

def test_create_searchable_model():
    arch = [0, 1, 2, 3]
    model = create_searchable_nano_u(
        input_shape=(48, 64, 3),
        filters=[16, 32, 64],
        bottleneck_filters=64,
        arch_seq=arch
    )
    assert isinstance(model, tf.keras.Model)
    assert model.name == "nas_nano_u"

def test_nas_search_loop(tmp_path):
    searcher = NASSearcher(
        input_shape=(48, 64, 3),
        filters=[16, 32, 64],
        bottleneck=64,
        population_size=2,
        generations=1,
        output_dir=str(tmp_path)
    )
    
    def dummy_train(model, epochs):
        class DummyHist:
            def __init__(self):
                self.history = {'accuracy': [0.5, 0.6], 'val_accuracy': [0.5, 0.55]}
        return DummyHist()
    
    results = searcher.search(dummy_train, None)
    
    assert "best_arch" in results
    assert len(results["history"]) == 1
    assert len(results["history"][0]) == 2 # Population size
    assert (Path(tmp_path) / "gen_0_results.csv").exists()

if __name__ == "__main__":
    pytest.main([__file__])
