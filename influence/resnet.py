from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import tensorflow as tf
import math
#adapted file given from TA to older tesnerflow version to avoid keras
from influence.genericNeuralNet4Resnet20 import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


class ResNet20(GenericNeuralNet):
    """ResNet-20 for CIFAR-10 following the influence library pattern."""

    def __init__(self, input_side, input_channels, conv_patch_size, hidden1_units, 
                 hidden2_units, hidden3_units, weight_decay, num_classes, batch_size,
                 data_sets, initial_learning_rate, damping, decay_epochs, mini_batch,
                 train_dir, log_dir, model_name, **kwargs):
        
        self.weight_decay = weight_decay
        self.input_side = input_side
        self.input_channels = input_channels
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.conv_patch_size = conv_patch_size
        # For ResNet, these represent the channel dimensions
        self.hidden1_units = hidden1_units  # 16 channels
        self.hidden2_units = hidden2_units  # 32 channels
        self.hidden3_units = hidden3_units  # 64 channels

        super(ResNet20, self).__init__(
            input_dim=self.input_dim,
            weight_decay=weight_decay,
            num_classes=num_classes,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            damping=damping,
            decay_epochs=decay_epochs,
            mini_batch=mini_batch,
            train_dir=train_dir,
            log_dir=log_dir,
            model_name=model_name,
            **kwargs
        )

    def batch_norm(self, x, name):
        """Batch normalization layer."""
        with tf.variable_scope(name):
            return tf.contrib.layers.batch_norm(
                x, 
                decay=0.9,
                epsilon=1e-5,
                center=True,
                scale=True,
                is_training=True,
                scope='bn'
            )

    def residual_block(self, x, in_channels, out_channels, stride, block_name):
        """Basic residual block."""
        with tf.variable_scope(block_name):
            # First convolution
            with tf.variable_scope('conv1'):
                W1 = variable_with_weight_decay(
                    'weights',
                    [3, 3, in_channels, out_channels],
                    stddev=math.sqrt(2.0 / (3 * 3 * in_channels)),
                    wd=self.weight_decay
                )
                conv1 = conv2d(x, W1, stride)
                bn1 = self.batch_norm(conv1, 'bn1')
                relu1 = tf.nn.relu(bn1)

            # Second convolution
            with tf.variable_scope('conv2'):
                W2 = variable_with_weight_decay(
                    'weights',
                    [3, 3, out_channels, out_channels],
                    stddev=math.sqrt(2.0 / (3 * 3 * out_channels)),
                    wd=self.weight_decay
                )
                conv2 = conv2d(relu1, W2, 1)
                bn2 = self.batch_norm(conv2, 'bn2')

            # Shortcut connection
            if stride != 1 or in_channels != out_channels:
                with tf.variable_scope('shortcut'):
                    W_shortcut = variable_with_weight_decay(
                        'weights',
                        [1, 1, in_channels, out_channels],
                        stddev=math.sqrt(2.0 / in_channels),
                        wd=self.weight_decay
                    )
                    shortcut = conv2d(x, W_shortcut, stride)
                    shortcut = self.batch_norm(shortcut, 'bn_shortcut')
            else:
                shortcut = x

            # Add and activate
            output = tf.nn.relu(bn2 + shortcut)
            return output

    def get_all_params(self):
        """Get all trainable parameters."""
        all_params = []
        for var in tf.trainable_variables():
            if 'weights' in var.name or 'biases' in var.name:
                all_params.append(var)
        return all_params

    def retrain(self, num_steps, feed_dict):        
        """Retrain the model."""
        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        """Create placeholder inputs."""
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def inference(self, input_x):        
        """Build the ResNet-20 model."""
        
        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        
        # Initial convolution
        with tf.variable_scope('conv1'):
            W_conv1 = variable_with_weight_decay(
                'weights',
                [3, 3, self.input_channels, 16],
                stddev=math.sqrt(2.0 / (3 * 3 * self.input_channels)),
                wd=self.weight_decay
            )
            conv1 = conv2d(input_reshaped, W_conv1, 1)
            bn1 = self.batch_norm(conv1, 'bn1')
            relu1 = tf.nn.relu(bn1)

        # Stack 1: 3 blocks with 16 filters
        block1_1 = self.residual_block(relu1, 16, 16, 1, 'block1_1')
        block1_2 = self.residual_block(block1_1, 16, 16, 1, 'block1_2')
        block1_3 = self.residual_block(block1_2, 16, 16, 1, 'block1_3')

        # Stack 2: 3 blocks with 32 filters (downsample)
        block2_1 = self.residual_block(block1_3, 16, 32, 2, 'block2_1')
        block2_2 = self.residual_block(block2_1, 32, 32, 1, 'block2_2')
        block2_3 = self.residual_block(block2_2, 32, 32, 1, 'block2_3')

        # Stack 3: 3 blocks with 64 filters (downsample)
        block3_1 = self.residual_block(block2_3, 32, 64, 2, 'block3_1')
        block3_2 = self.residual_block(block3_1, 64, 64, 1, 'block3_2')
        block3_3 = self.residual_block(block3_2, 64, 64, 1, 'block3_3')

        # Global average pooling
        gap = tf.reduce_mean(block3_3, axis=[1, 2])

        # Fully connected layer
        with tf.variable_scope('fc'):
            W_fc = variable_with_weight_decay(
                'weights',
                [64, self.num_classes],
                stddev=math.sqrt(2.0 / 64),
                wd=self.weight_decay
            )
            b_fc = variable(
                'biases',
                [self.num_classes],
                tf.constant_initializer(0.0)
            )
            logits = tf.matmul(gap, W_fc) + b_fc

        return logits

    def predictions(self, logits):
        """Get predictions from logits."""
        preds = tf.nn.softmax(logits, name='preds')
        return preds