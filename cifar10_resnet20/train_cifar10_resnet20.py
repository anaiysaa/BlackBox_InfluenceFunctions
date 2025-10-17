from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf

print("=" * 60)
print("CIFAR-10 ResNet20 Training Script")
print("=" * 60)

import influence.experiments as experiments
from influence.resnet import ResNet20

from load_cifar10 import load_small_cifar10, load_cifar10

print("\n[1/5] Loading CIFAR-10 dataset...")
data_sets = load_small_cifar10('data')
print(f"✓ Dataset loaded successfully")
print(f"  - Training samples: {len(data_sets.train.labels)}")
print(f"  - Validation samples: {len(data_sets.validation.labels)}")
print(f"  - Test samples: {len(data_sets.test.labels)}")    

print("\n[2/5] Setting up model parameters...")
num_classes = 10
input_side = 32
input_channels = 3
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
# ResNet doesn't use these hidden unit parameters, but keeping for compatibility
hidden1_units = 16
hidden2_units = 32
hidden3_units = 64
conv_patch_size = 3
keep_probs = [1.0, 1.0]

print(f"✓ Parameters configured")
print(f"  - Input: {input_side}x{input_side}x{input_channels}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {initial_learning_rate}")
print(f"  - Weight decay: {weight_decay}")

print("\n[3/5] Initializing ResNet20 model...")
model = ResNet20(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    model_name='cifar10_small_resnet20')
print(f"✓ Model initialized: cifar10_small_resnet20")

print("\n[4/5] Starting training...")
print(f"  - Total steps: 500,000")
print(f"  - This may take several hours...")
num_steps = 500000
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1
print(f"✓ Training completed at step {iter_to_load}")

print("\n[5/5] Running influence function retraining test...")
test_idx = 6558
print(f"  - Test index: {test_idx}")
print(f"  - Removing 100 most influential training points")
print(f"  - Retraining for 30,000 steps...")

actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model, 
    test_idx=test_idx, 
    iter_to_load=iter_to_load, 
    num_to_remove=100,
    num_steps=30000, 
    remove_type='maxinf',
    force_refresh=True)
print(f"✓ Retraining test completed")

print("\n[6/6] Saving results...")
output_file = 'output/cifar10_small_resnet20_iter-500k_retraining-100.npz'
np.savez(
    output_file, 
    actual_loss_diffs=actual_loss_diffs, 
    predicted_loss_diffs=predicted_loss_diffs, 
    indices_to_remove=indices_to_remove
    )
print(f"✓ Results saved to: {output_file}")

print("\n" + "=" * 60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 60)