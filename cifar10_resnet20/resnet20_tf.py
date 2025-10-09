from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import tensorflow as tf
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from tensorflow.contrib.learn.python.learn.datasets import base


class ResNet20(GenericNeuralNet):
    """
    ResNet20 implementation compatible with influence functions library.
    """
    
    def __init__(self, input_side, input_channels, num_classes, 
                 batch_size, data_sets, weight_decay, 
                 initial_learning_rate, train_dir='output', 
                 log_dir='log', model_name='resnet20', **kwargs):
        
        self.input_side = input_side
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        
        super(ResNet20, self).__init__(
            input_dim=input_side * input_side * input_channels,
            weight_decay=weight_decay,
            num_classes=num_classes,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            keep_probs=None,
            decay_epochs=[40000, 60000],
            mini_batch=True,
            train_dir=train_dir,
            log_dir=log_dir,
            model_name=model_name,
            **kwargs
        )
    
    def placeholder_inputs(self):
        """Generate placeholder variables for input tensors."""
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_side, self.input_side, self.input_channels),
            name='input_placeholder'
        )
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder'
        )
        return input_placeholder, labels_placeholder
    
    def _conv_bn_relu(self, x, filters, kernel_size, stride, name, training):
        """Conv -> BatchNorm -> ReLU block."""
        with tf.variable_scope(name):
            # Convolution
            in_channels = x.get_shape().as_list()[-1]
            kernel = variable_with_weight_decay(
                'weights',
                shape=[kernel_size, kernel_size, in_channels, filters],
                stddev=np.sqrt(2.0 / (kernel_size * kernel_size * in_channels)),
                wd=self.weight_decay
            )
            conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')
            
            # Batch normalization (without tracking running stats)
            gamma = variable('gamma', [filters], tf.constant_initializer(1.0))
            beta = variable('beta', [filters], tf.constant_initializer(0.0))
            batch_mean, batch_var = tf.nn.moments(conv, axes=[0, 1, 2], keep_dims=False)
            bn = tf.nn.batch_normalization(conv, batch_mean, batch_var, beta, gamma, 1e-5)
            
            # ReLU
            out = tf.nn.relu(bn)
            return out
    
    def _conv_bn(self, x, filters, kernel_size, stride, name, training):
        """Conv -> BatchNorm block (no activation)."""
        with tf.variable_scope(name):
            # Convolution
            in_channels = x.get_shape().as_list()[-1]
            kernel = variable_with_weight_decay(
                'weights',
                shape=[kernel_size, kernel_size, in_channels, filters],
                stddev=np.sqrt(2.0 / (kernel_size * kernel_size * in_channels)),
                wd=self.weight_decay
            )
            conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')
            
            # Batch normalization
            gamma = variable('gamma', [filters], tf.constant_initializer(1.0))
            beta = variable('beta', [filters], tf.constant_initializer(0.0))
            batch_mean, batch_var = tf.nn.moments(conv, axes=[0, 1, 2], keep_dims=False)
            bn = tf.nn.batch_normalization(conv, batch_mean, batch_var, beta, gamma, 1e-5)
            
            return bn
    
    def _basic_block(self, x, filters, stride, name, training):
        """Basic residual block: two 3x3 convs with residual connection."""
        with tf.variable_scope(name):
            in_channels = x.get_shape().as_list()[-1]
            
            # First conv: 3x3, stride=stride
            out = self._conv_bn_relu(x, filters, 3, stride, 'conv1', training)
            
            # Second conv: 3x3, stride=1 (no activation yet)
            out = self._conv_bn(out, filters, 3, 1, 'conv2', training)
            
            # Shortcut connection
            if stride != 1 or in_channels != filters:
                # Option B: 1x1 convolution projection
                shortcut = self._conv_bn(x, filters, 1, stride, 'shortcut', training)
            else:
                shortcut = x
            
            # Add residual and apply ReLU
            out = tf.nn.relu(out + shortcut)
            return out
    
    def _make_layer(self, x, filters, num_blocks, stride, name, training):
        """Stack of residual blocks."""
        with tf.variable_scope(name):
            # First block with specified stride
            out = self._basic_block(x, filters, stride, 'block_0', training)
            # Remaining blocks with stride=1
            for i in range(1, num_blocks):
                out = self._basic_block(out, filters, 1, 'block_{}'.format(i), training)
            return out
    
    def inference(self, input_placeholder, training=True):
        """
        Build the ResNet20 model.
        ResNet20 = ResNet(BasicBlock, [3, 3, 3])
        """
        with tf.variable_scope('resnet20'):
            # Initial conv: 3x3, 16 filters, stride=1
            net = self._conv_bn_relu(input_placeholder, 16, 3, 1, 'conv1', training)
            
            # Layer 1: 3 blocks, 16 filters, stride=1
            net = self._make_layer(net, 16, 3, 1, 'layer1', training)
            
            # Layer 2: 3 blocks, 32 filters, stride=2
            net = self._make_layer(net, 32, 3, 2, 'layer2', training)
            
            # Layer 3: 3 blocks, 64 filters, stride=2
            net = self._make_layer(net, 64, 3, 2, 'layer3', training)
            
            # Global average pooling - use reduce_mean for gradient support
            net = tf.reduce_mean(net, axis=[1, 2], keep_dims=False)
            
            # Fully connected layer
            with tf.variable_scope('fc'):
                w = variable_with_weight_decay(
                    'weights',
                    shape=[64, self.num_classes],
                    stddev=np.sqrt(2.0 / 64),
                    wd=self.weight_decay
                )
                b = variable('biases', [self.num_classes], tf.constant_initializer(0.0))
                logits = tf.matmul(net, w) + b
            
            return logits
    
    def predictions(self, logits):
        """Return the predicted class probabilities."""
        preds = tf.nn.softmax(logits, name='preds')
        return preds