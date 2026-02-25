import tensorflow as tf
from src.nas import compute_layer_redundancy

def run_test():
    # Test case 1: 32x64 with rank 32 (wide matrix)
    print("--- Test 1: Wide Matrix (32x64), Rank 32 ---")
    data = tf.random.normal((32, 64))
    res = compute_layer_redundancy(data)
    print(res)

    # Test case 2: 64x32 with rank 32 (tall matrix)
    print("\n--- Test 2: Tall Matrix (64x32), Rank 32 ---")
    data = tf.random.normal((64, 32))
    res = compute_layer_redundancy(data)
    print(res)
    
    # Test case 3: 4D tensor (Batch=16, H=24, W=32, C=64)
    print("\n--- Test 3: 4D Tensor (16, 24, 32, 64) ---")
    data = tf.random.normal((16, 24, 32, 64))
    res = compute_layer_redundancy(data)
    print(res)

    # Test case 4: Rank deficient matrix
    print("\n--- Test 4: Rank Deficient (32x64), Rank 1 ---")
    perfect_corr = tf.tile(tf.random.normal((32, 1)), [1, 64])
    res = compute_layer_redundancy(perfect_corr)
    print(res)

run_test()
