from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

import influence.experiments as experiments
from influence.resnet import ResNet20
from load_cifar10 import load_cifar10, load_small_cifar10

# Load dataset
data_sets = load_cifar10('data')

# Model hyperparameters
num_classes = 10
input_side = 32
input_channels = 3
batch_size = 128
weight_decay = 0.0001
initial_learning_rate = 0.1
num_steps = 50000

print("Initializing ResNet20 model...")
model = ResNet20(
    input_side=input_side,
    input_channels=input_channels,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    weight_decay=weight_decay,
    initial_learning_rate=initial_learning_rate,
    train_dir='output/cifar10',
    log_dir='log/cifar10',
    model_name='cifar10_resnet20'
)
print("Model initialized successfully!")

print("Starting training for {} steps...".format(num_steps))
model.train(
    num_steps=num_steps,
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000
)
print("Training finished!")

# Test influence functions
iter_to_load = num_steps - 1
test_idx = 0

print("Starting influence retraining test on test index {}...".format(test_idx))
actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model,
    test_idx=test_idx,
    iter_to_load=iter_to_load,
    num_to_remove=100,
    num_steps=30000,
    remove_type='maxinf',
    force_refresh=True
)
print("Influence retraining test finished!")

# Save results
save_path = 'output/cifar10_resnet20_iter-50k_retraining-100.npz'
np.savez(
    save_path,
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs=predicted_loss_diffs,
    indices_to_remove=indices_to_remove
)
print("Results saved to {}".format(save_path))