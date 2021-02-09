Version 1.0
07 December 2020

This software allows the reproduction of machine learning models used in:
J.S. Hyatt and M.S. Lee. Multi-modal generative adversarial networks make realistic and diverse but untrustworthy predictions when applied to ill-posed problems. In /Proceedings of the Workshop on Artificial Intelligence Safety, co-located with the 35th AAAI Conference on Artificial Intelligence/ (2021).

MOTIVATION
To allow the reproduction of results from our paper. Part of the motivation for the paper itself was the critical examination of work other people have done, and what can be done to improve similar work in the future. In the same vein, we welcome that type of feedback from others, especially if it leads to better science and safer machine learning outcomes in the future.

DESCRIPTION
We do not include code to perform different tests of latent space reproduction, loss analysis, distribution matching, etc. There are many different possible such tests, and for the most part they can be reproduced with a few lines of python code. 

The code itself is broken into four scripts. One is used to convert the data to a TFRecord file format for optimal use by TensorFlow. The remaining three build the dataset and models and define training steps and the full custom training loop, and perform the actual training. Details are included in the paper and documented in the code.

During training, the code generates one .hdf5 model file for each of the generator, encoder, and critic models, as well as the training loss and validation loss over time. These files are all updated continuously during training. It also generates checkpoints every 10 epochs so that training can be easily resumed from any point. Optionally, it may also generate demo images from each class continuously during training, for use in generating a gif of the training.

Finally, the output of the custom training loop is continuously printed during training as a table of losses, so it can be monitored during training. This also indicates progress during training.

The four scripts and their functions are:

convert_npy_to_tfrecords.py
-Downloads the .npy format dataset using built-in TensorFlow function and converts it to TFRecord.
-Contains a function for verifying the TFRecord contains the correct information.
-Sets aside some examples for later demo use, like making gifs.

base_functions.py
(called by convert_npy_to_tfrecords.py, models.py, and main.py)
Contains functions for:
-Adding annealed instance noise to the data.
-Defining the model building blocks (from convolution blocks to ResNeXt).
-Rescaling the data.
-Making the TF Dataset object, including downsampling, latent vector sampling, caching, shuffling, batching, and prefetching.
-Defining the loss functions used to train the networks.

models.py
(called by main.py)
Contains functions for building the generator, encoder, and critic.

main.py
Contains functions for:
-Making the training and validation datasets.
-Making the demo dataset.
-Building the neural networks.
-Defining the training steps for each of the three models.
-Evaluating the performance of each of the three models.
-Generating images by passing the demo dataset through the models.
-Defining the training loop.
-Performing training by running the training loop.

GETTING STARTED
These instructions will explain how to get this code running.

PREREQUISITES
To run this software, you need an installation of Python 3. We recommend Anaconda to properly manage packages. Required packages include TensorFlow 2.2 or higher and numpy. Matplotlib is not required in the code, but will be required to visualize results.

INSTALLING AND CONFIGURATION
Make sure that all four .py files are in your working directory or in a path python will search.

RUNNING THE SOFTWARE
1. Making the dataset
In convert_npy_to_tfrecords.py, set the filepath you would like to save the dataset to, run the file, and then execute the command make_TFRecords(). This will download the fashion-MNIST training and validation data, minimally preprocess it, and save it as TFRecords files at the indicated filepath, as well as saving a demo dataset for producing gifs (containing one example from each class) as a pair of .npy files. This code calls base_functions.py.

2. Build the dataset object, build the models, define the custom training loop, and train the models
In main.py, set hyperparameters as desired. These are all towards the beginning of the script and accompanied by commented documentation. See the documentation for details (there are too many to list here). Then simply run main.py. This code calls base_functions.py and models.py.

PUBLIC DOMAIN DEDICATION
The text of the CC0 license can be found [here - hyperlink to https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt]. A local copy of the license is in the file LICENSE.txt.