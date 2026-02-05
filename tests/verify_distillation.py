import os
import sys
import tensorflow as tf
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_with_distillation
from src.models import create_nano_u, create_bu_net

def verify_distillation():
    print("Running distillation verification...")
    
    # Create models
    student = create_nano_u(input_shape=(48, 64, 3))
    teacher = create_bu_net(input_shape=(48, 64, 3))
    
    # Create synthetic data
    x = np.random.rand(4, 48, 64, 3).astype(np.float32)
    y = np.random.randint(0, 2, (4, 48, 64, 1)).astype(np.float32)
    
    config = {
        'epochs': 2,
        'batch_size': 2,
        'alpha': 0.5,
        'temperature': 2.0,
        'learning_rate': 0.001,
        'output_dir': 'results/test_verify'
    }
    
    if not os.path.exists('results/test_verify'):
        os.makedirs('results/test_verify')
        
    history = train_with_distillation(student, teacher, config, (x, y), (x, y))
    
    distill_losses = history['distillation_loss']
    print(f"Distillation losses: {distill_losses}")
    
    # Assert that distillation loss is greater than 0
    for loss in distill_losses:
        if loss <= 0:
            print(f"❌ Verification failed: Distillation loss is {loss}")
            sys.exit(1)
            
    print("✅ Verification successful: Distillation loss is positive.")

if __name__ == "__main__":
    verify_distillation()
