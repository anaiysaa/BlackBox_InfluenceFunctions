# Adapted from load_mnist.py for CIFAR-10 dataset

import pickle
import numpy as np
import os
import tarfile

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet


def unpickle(file):
    """Load pickled CIFAR-10 data file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def extract_cifar10_images_and_labels(batch_file):
    """Extract images and labels from a single CIFAR-10 batch file.
    
    Args:
        batch_file: Path to a CIFAR-10 batch file.
        
    Returns:
        images: numpy array of images.
        labels: numpy array of labels.
    """
    print('Extracting', batch_file)
    batch_dict = unpickle(batch_file)
    images = batch_dict['data']
    labels = batch_dict['labels']
    
    # Reshape from (N, 3072) to (N, 32, 32, 3)
    # CIFAR-10 stores as [red_channel, green_channel, blue_channel] flattened
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(labels)
    
    return images, labels


def load_cifar10(train_dir, validation_size=5000):
    """Load CIFAR-10 dataset.
    
    Args:
        train_dir: Directory where CIFAR-10 data is stored or will be downloaded.
        validation_size: Number of training examples to use for validation.
        
    Returns:
        base.Datasets object containing train, validation, and test datasets.
    """
    SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/'
    CIFAR10_FILE = 'cifar-10-python.tar.gz'
    
    # Download if not exists
    local_file = base.maybe_download(CIFAR10_FILE, train_dir, SOURCE_URL + CIFAR10_FILE)
    
    # Extract if not already extracted
    cifar_dir = os.path.join(train_dir, 'cifar-10-batches-py')
    if not os.path.exists(cifar_dir):
        print('Extracting CIFAR-10 archive...')
        with tarfile.open(local_file, 'r:gz') as tar:
            tar.extractall(path=train_dir)
    
    # Load training data from 5 batches
    train_images_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, 'data_batch_{}'.format(i))
        images, labels = extract_cifar10_images_and_labels(batch_file)
        train_images_list.append(images)
        train_labels_list.append(labels)
    
    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    
    # Load test data
    test_batch_file = os.path.join(cifar_dir, 'test_batch')
    test_images, test_labels = extract_cifar10_images_and_labels(test_batch_file)
    
    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))
    
    # Split validation set
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    
    # Normalize to [0, 1]
    train_images = train_images.astype(np.float32) / 255
    validation_images = validation_images.astype(np.float32) / 255
    test_images = test_images.astype(np.float32) / 255
    
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    
    return base.Datasets(train=train, validation=validation, test=test)


def load_small_cifar10(train_dir, validation_size=5000, random_seed=0):
    """Load a smaller subset of CIFAR-10 (1/10 of training data).
    
    Args:
        train_dir: Directory where CIFAR-10 data is stored or will be downloaded.
        validation_size: Number of training examples to use for validation.
        random_seed: Random seed for reproducibility.
        
    Returns:
        base.Datasets object containing train, validation, and test datasets.
    """
    np.random.seed(random_seed)
    data_sets = load_cifar10(train_dir, validation_size)
    
    train_images = data_sets.train.x
    train_labels = data_sets.train.labels
    perm = np.arange(len(train_labels))
    np.random.shuffle(perm)
    num_to_keep = int(len(train_labels) / 10)
    perm = perm[:num_to_keep]
    train_images = train_images[perm, :]
    train_labels = train_labels[perm]
    
    validation_images = data_sets.validation.x
    validation_labels = data_sets.validation.labels
    # Not reducing validation set size (same as MNIST)
    
    test_images = data_sets.test.x
    test_labels = data_sets.test.labels
    # Not reducing test set size (same as MNIST)
    
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)
    
    return base.Datasets(train=train, validation=validation, test=test)