from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import tensorflow as tf
import os
import sys
import influence.experiments4Resnet20 as experiments
from influence.resnet import ResNet20
from load_cifar10 import load_small_cifar10
from influence.genericNeuralNet4Resnet20 import GenericNeuralNet

print("========== CIFAR-10 ResNet20 Stable Training Script ==========\n")

# --------------------------------------------------------------
# [0] Monkey-patch fix for vectorization (keeps params consistent)
# --------------------------------------------------------------
def fixed_get_vec_to_list_fn(self):
  params_val = self.sess.run(self.params)
  flattened_params = [p.flatten() for p in params_val]
  self.num_params = sum(p.size for p in params_val)
  print('Total number of parameters:', self.num_params)

  def vec_to_list(v):
    return_list = []
    cur_pos = 0
    for p in params_val:
      p_flat = p.flatten()
      return_list.append(v[cur_pos:cur_pos + len(p_flat)].reshape(p.shape))
      cur_pos += len(p_flat)
    assert cur_pos == len(v), "Vector size mismatch"
    return return_list

  return vec_to_list

GenericNeuralNet.get_vec_to_list_fn = fixed_get_vec_to_list_fn


# --------------------------------------------------------------
# [1] Load CIFAR-10 dataset
# --------------------------------------------------------------
print("[1/6] Loading CIFAR-10 dataset...")
data_sets = load_small_cifar10('data')
print("Dataset loaded successfully.\n")


# --------------------------------------------------------------
# [2] Model parameters
# --------------------------------------------------------------
print("[2/6] Setting model parameters...")

num_classes = 10
input_side = 32
input_channels = 3
weight_decay = 0.001
batch_size = 250

initial_learning_rate = 1e-4
damping = 0.01
decay_epochs = [15000, 30000]

hidden1_units = 16
hidden2_units = 32
hidden3_units = 64
conv_patch_size = 3
keep_probs = [1.0, 1.0]

print("Model parameters configured.\n")


# --------------------------------------------------------------
# [3] Initialize ResNet20
# --------------------------------------------------------------
print("[3/6] Initializing ResNet20 model...")

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
    damping=damping,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    model_name='cifar10_small_resnet20_stable'
)
print("Model initialized successfully.\n")


# --------------------------------------------------------------
# [4] Train ResNet20 with auto-save checkpoints
# --------------------------------------------------------------
# print("[4/6] skipping training phase...")

# num_steps = 500000
# save_every = 10000
checkpoint_dir = 'output/checkpoints'

# if not os.path.exists(checkpoint_dir):
#   os.makedirs(checkpoint_dir)

# for step_block in range(0, num_steps, save_every):
#   model.train(
#     num_steps=save_every,
#     iter_to_switch_to_batch=10000000,
#     iter_to_switch_to_sgd=10000000
#   )
#   checkpoint_path = os.path.join(checkpoint_dir, "resnet20_step_{}.ckpt".format(step_block + save_every))
#   saver = tf.train.Saver()
#   saver.save(model.sess, checkpoint_path)
#   print("Checkpoint saved: {}\n".format(checkpoint_path))
# iter_to_load = num_steps - 1
# print("Training completed successfully.\n")

print("[4/6] Loading pretrained checkpoint...")
# checkpoint_path = "output/cifar10_small_resnet20_stable-checkpoint-700000"

# # Make sure the checkpoint exists
# if not os.path.exists(checkpoint_path + '.index'):
#     raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

# # Directly restore TensorFlow variables
# saver = tf.train.Saver()
# saver.restore(model.sess, checkpoint_path)
# print("[4/6] Checkpoint loaded successfully.\n")
# print("[4/6] continuing training to 1,000,000\n")

# # Continue training
# start_step = 500000
# end_step = 1000000
# save_every = 100000

# for step_block in range(start_step, end_step, save_every):
#     steps_to_train = min(save_every, end_step - step_block)
#     model.train(num_steps=steps_to_train,
#                 iter_to_switch_to_batch=10000000,
#                 iter_to_switch_to_sgd=10000000)
#     new_checkpoint_path = os.path.join(checkpoint_dir, "resnet20_step_{}.ckpt".format(step_block + steps_to_train))
#     saver.save(model.sess, new_checkpoint_path)
#     print("Checkpoint saved: {}\n".format(new_checkpoint_path))

# print("Training continued to 1,000,000 steps.\n")

# --------------------------------------------------------------
# [5] Influence retraining test
# --------------------------------------------------------------

print("[5/6] Running influence function retraining test...")
test_idx = 6558

# Use low LR to avoid divergence
model.learning_rate = 1e-5

print("Removing 5 most influential training points")
#og damping 2.0
print("Retraining...\n")

actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model,
    test_idx=test_idx,
    iter_to_load = 500000,
    num_to_remove=10,
    num_steps= 50,
    remove_type='maxinf',
    force_refresh=False
)

print("\nRetraining test completed successfully.\n")


# --------------------------------------------------------------
# [6] Save final results
# --------------------------------------------------------------
print("[6/6] Saving results...")
output_file = 'output/cifar10_small_resnet20_stable_retrain_results.npz'

np.savez(
    output_file,
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs=predicted_loss_diffs,
    indices_to_remove=indices_to_remove
)
print("\n========== ALL TASKS COMPLETED SUCCESSFULLY ==========")
