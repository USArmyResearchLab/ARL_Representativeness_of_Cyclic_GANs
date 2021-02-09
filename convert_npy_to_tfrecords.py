#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:07:33 2020

@author: student
"""

import numpy as np
import tensorflow as tf
from base_functions import one_one

# File path for saving the tfrecords files
path = '/home/student/Work/Keras/GAN/mySRGAN/new_implementation/multiGPU_datasetAPI/custom_train_loop_function_method/DATA_FOR_AAAI/code/'

###############################################################################

# Documentation:
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://www.tensorflow.org/guide/data

# How to shard a large dataset:
#   https://jkjung-avt.github.io/tfrecords-for-keras/
#   https://github.com/jkjung-avt/keras_imagenet
#   https://github.com/jkjung-avt/keras_imagenet/blob/master/data/build_imagenet_data.py <- this one builds the dataset

# Working example (partially adapted to produce this code):
#   https://stackoverflow.com/questions/57717004/tensorflow-modern-way-to-load-large-data

# Another working example:
#   https://stackoverflow.com/questions/50955798/keras-model-fit-with-tf-dataset-api-validation-data#50979587

# Some benefits to working with tf.data.Dataset as opposed to a numpy array:
#   https://stackoverflow.com/questions/47732186/tensorflow-tf-record-too-large-to-be-loaded-into-an-np-array-at-once?rq=1

###############################################################################

"""
First, download the fashion-MNIST dataset, preprocess, and convert to numpy.
"""

# Load training and validation (test) data using built-in TF function.
((x_train,
  y_train),
 (x_val,
  y_val)) = tf.keras.datasets.fashion_mnist.load_data()

# Expand dims to include depth dimension
x_train = np.expand_dims(x_train,
                         -1)
x_val = np.expand_dims(x_val,
                       -1)

# Convert from uint8 to float32
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

# By default, the intensities are in [0,255]. Rescale to [-1,1]. Do this SEPARATELY for the training and validation sets.
x_train = one_one(x_train)
x_val = one_one(x_val)

# Set aside the first example from each class as a "demo" dataset, for tracking changes in the model outputs during training.
indices = [19,2,1,13,6,8,4,9,18,0]
x_demo = np.zeros((10,28,28,1))
for i in range(10):
    x_demo[i] = x_val[indices[i]]
# Make a set of 4 "demo" z vectors.
# They will be very long, and can be adapted to any reasonable z_dim by simply chopping off the extra dimensions.
z_demo = np.random.randn(4,1000,1)

# The x data have now been preprocessed for conversion to tfrecords files.
# We are not doing anything with the class labels, but for the sake of completeness, here is the conversion:
y_train = tf.keras.utils.to_categorical(y_train,
                                        num_classes=10,
                                        dtype='float32')
y_val = tf.keras.utils.to_categorical(y_val,
                                      num_classes=10,
                                      dtype='float32')

###############################################################################

"""
FUNCTION DEFINITIONS
"""

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float (32 or 64 precision)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Returns an int64_list from bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_img_to_str(img):
    """
    Convert an image in numpy array format, shape (1, height, width, depth), to a string. Depth is 1 for grayscale, 3 for RGB.
    Returns:
        img_string
        height
        width
        depth
    """

    # height, width, and depth are all going to be features in the serialized example (just the same as the class label will be).
    height = img.shape[1]
    width = img.shape[2]
    depth = img.shape[3]

    img_string = img.tostring()

    return img_string, height, width, depth

def _convert_1hot_to_str(one_hot):
    """
    Convert a one-hot class label vector, shape (1, num_classes), to a string.
    Returns:
        one_hot_string
    """
    one_hot_string = one_hot.tostring()
    return one_hot_string

def _convert_to_example(img_string,
                        height,
                        width,
                        depth,
                        one_hot_string):
    """
    Serialize an example with a single image and corresponding one-hot class label.
    This creates a tf.Example message ready to be written to a file.

    Args:
        img_string:  string, image in array of shape (1, height, width, depth).
        height:  integer, image height in pixels.
        width:  integer, image width in pixels.
        depth:  integer, image depth in pixels.
        one_hot_string:  string, one-hot vector that identifies the ground truth label.
    Returns:
        Example proto
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {'img': _bytes_feature(img_string),
               'height': _int64_feature(height),
               'width': _int64_feature(width),
               'depth': _int64_feature(depth),
               'label': _bytes_feature(one_hot_string)}

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto

def _make_TFRecord(x,
                   y,
                   output_file):
    """
    Processes and saves an array of image data and array of label data as a TFRecord.

    Args:
        x:  numpy array of images.
            Has shape (num_examples, height, width, depth).
        y:  numpy array of one-hot class labels.
            Has shape (num_examples, num_classes)
        output_file:  string, filename for the saved TFRecord file.
    """

    # Open the file writer.  All of the examples will be written here.
    writer = tf.io.TFRecordWriter(output_file)

    for i in range(x.shape[0]):

        # Get the image and corresponding label for one example.
        img = x[i:i+1]
        label = y[i:i+1]

        # Convert the image to a string and obtain the image dimensions.
        (img_string,
         height,
         width,
         depth) = _convert_img_to_str(img)

        # Convert the one-hot class label to a string.
        one_hot_string = _convert_1hot_to_str(label)

        # Put all the features into the example.
        example = _convert_to_example(img_string,
                                      height,
                                      width,
                                      depth,
                                      one_hot_string)

        # Write the example into the TFRecords file.
        writer.write(example.SerializeToString())
        print(f'Created example {i} of {x.shape[0]}.')

    writer.close()
    print(f'Saved TFRecord to {output_file}.')

def _parse_example(serialized_example):
    """
    Takes a single serialized example and converts it back into an image (AS A TENSOR OBJECT) and the corresponding label.
    """

    features = {'img': tf.io.FixedLenFeature([],
                                             tf.string),
                'height': tf.io.FixedLenFeature([],
                                                tf.int64),
                'width': tf.io.FixedLenFeature([],
                                               tf.int64),
                'depth': tf.io.FixedLenFeature([],
                                               tf.int64),
                'label': tf.io.FixedLenFeature([],
                                               tf.string)}

    parsed_example = tf.io.parse_single_example(serialized=serialized_example,
                                                features=features)

    # Get the class label.
    label = parsed_example['label']
    label = tf.io.decode_raw(label,
                             tf.float32)

    # Get the image dimensions.
    height = parsed_example['height']
    width = parsed_example['width']
    depth = parsed_example['depth']

    # Get the raw byte string and convert it to a tensor object.
    img = parsed_example['img']
    img = tf.io.decode_raw(img,
                           tf.float32)
    # The tensor object is 1-dimensional, so reshape it using the image dimensions we obtained earlier.
    img = tf.reshape(img,
                     shape=(height,
                            width,
                            depth))

    return img, label

###############################################################################

def make_TFRecords():

    """
    Function that creates training and validation tfrecords files using the above functions.
    """

    _make_TFRecord(x_train,
                   y_train,
                   path + 'xy_train_fMNIST.tfrecords')

    _make_TFRecord(x_val,
                   y_val,
                   path + 'xy_val_fMNIST.tfrecords')

    np.save(path + 'x_demo.npy',
            x_demo)
    np.save(path + 'z_demo.npy',
            z_demo)

def verify_TFRecords(path):

    """
    Function that verifies that the tfrecords files were created properly.

    Args:
        path: location of xy_train.tfrecords and xy_val.tfrecords

    Plots several images from the validation set and prints the corresponding labels.
    """

    import matplotlib.pyplot as plt

    xy_val = path + 'xy_val_fMNIST.tfrecords'

    xy_val = tf.data.TFRecordDataset(xy_val)

    # Check the first 5 examples from the dataset.
    xy_val = xy_val.take(5).map(_parse_example)

    plt.close('all')

    # Each example looks like (img, label)
    for ex in xy_val:
        img = ex[0].numpy()
        label = ex[1].numpy()

        plt.figure()
        plt.imshow(img[:,:,0],
                   cmap='gray',
                   vmin=-1,
                   vmax=1)
        print(label)