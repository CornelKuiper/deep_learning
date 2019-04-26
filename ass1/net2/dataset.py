import tensorflow as tf
import numpy as np
from read import _parse_function


def get_dataset(batch_size, dataset_name):
    if dataset_name == 'fashion_mnist':
        dataset = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)
    elif dataset_name == 'cifar10':
        dataset = tf.keras.datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(_parse_function)
    train_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=500))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    train_dataset = train_dataset.repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(batch_size).repeat()

    return train_dataset, test_dataset
