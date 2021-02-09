#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:54:57 2020
"""

###############################################################################

"""
MODULES
"""

import tensorflow as tf

#from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import (Add,
                                     BatchNormalization,
                                     Concatenate,
                                     Convolution2D,
                                     Conv2DTranspose,
                                     Lambda,
                                     LayerNormalization,
                                     LeakyReLU,
                                     MaxPooling2D)

###############################################################################

"""
LIST OF FUNCTIONS

instance_noise_alpha
instance_noise_B_Bdomain
instance_noise_R_Bdomain
instance_noise_R_Rdomain
instance_noise_B_Rdomain

conv
tconv

add_common_layers
grouped_convolution
residual_block

one_one

make_dataset

down
up

KL_batchwise_loss
KL_variational_loss
reparameterize

wasserstein_loss
RandomWeightedAverage

L2_magnitude
generator_loss
encoder_loss
encoder_loss_var
critic_loss
"""

###############################################################################

"""
FUNCTION DETERMINES INSTANCE NOISE AMPLITUDE
"""

@tf.function
def instance_noise_alpha(epoch,
                         cutoff_epoch):

    # if epoch < cutoff_epoch, add noise.
    # else: do not add noise.
    alpha = tf.cond(tf.less(epoch, # condition
                            cutoff_epoch),
                    lambda: tf.divide(epoch, # returned value if condition
                                      cutoff_epoch),
                    lambda: tf.identity(1.0)) # returned value if not condition

    return alpha

"""
FUNCTIONS FOR APPLYING INSTANCE NOISE

The purpose of instance noise is to force the inputs of C to have the same support throughout training, even at the beginning of training when G is not able to produce realistic outputs.
    E does not need to be fed noisy inputs, since it should still be able to provide useful gradients even if the real and fake images do not have the same support. However, there is no additional cost to indiscriminately add noise to all images, even those only input into E. All models have at most some loss terms that they might be able to optimize during early training, if C is receiving noisy inputs but E is not, but this optimization would only be possible if G was already trained. We therefore add noise to all images.

Derivation:
    1) Need to apply noise to same domain throughout entire problem, and
    2) The domain to which it is applied should be the domain the critic operates on, since that is the domain where support needs to be the same between the real and fake submissions.

    Applying noise to B when B is the input to C is not the same as applying noise to B when R is the input to C; likewise for applying noise to R.

Notation: B is an image, and R is the residual B - A, where A is the low-resolution downsampling of B. B' and R' are the noise-ified B and R, respectively.

if critic_input == 'B'
or critic_input == 'AB':
    Applying noise to B:
    R = B - A
 -> B = R + A
    B' = R' + A
       = a * B + (1 - a) * noise
 -> R' = a * (R + A) + (1 - a) * noise - A
       = a * R + (1 - a) * (noise - A)

elif critic_input == 'R':
    Alternatively, applying noise to R:
    R' = B' - A
       = a * R + (1 - a) * noise
 -> B' = a * (B - A) + (1 - a) * noise + A
       = a * B + (1 - a) * (noise + A)
"""

@tf.function
def instance_noise_B_Bdomain(B,
                             alpha):

    """
    Args:
        B: A high-resolution image (real or generated).
        alpha: The fraction of B that goes into the output (the remainder is uniform noise).

    Returns:
        if alpha < 1:
            B_noise: alpha * B + (1 - alpha) * noise
            noise = random.uniform(-1,1)
        else:
            B
    """

    B_noisy = tf.cond(tf.less(alpha,
                              1),
                      # if alpha < 1:
                      # return B, noise-ified in the B domain
                      lambda: alpha * B + \
                              (tf.identity(1.0) - alpha) * \
                              K.random_uniform(shape=tf.shape(B),
                                               minval=-1,
                                               maxval=1),
                      # else:
                      # return B with no noise-ification
                      lambda: B)

    return B_noisy

@tf.function
def instance_noise_R_Bdomain(R,
                             A,
                             alpha):

    """
    Args:
        R: A residual, R = B - A (real or generated).
        A: The conditioning image, in this case A = up(down(B)).
        alpha: The noise fraction in the B domain.

    Returns:
        if alpha < 1:
            R_noise: alpha * R + (1 - alpha) * (noise - A)
            noise = random.uniform(-1,1)
        else:
            R
    """

    R_noisy = tf.cond(tf.less(alpha,
                              1),
                      # if alpha < 1:
                      # return R, noise-ified in the B domain
                      lambda: alpha * R + \
                              (tf.identity(1.0) - alpha) * \
                              (K.random_uniform(shape=tf.shape(R),
                                                minval=-1,
                                                maxval=1) - A),
                      # else:
                      # return R with no noise-ification
                      lambda: R)

    return R_noisy

@tf.function
def instance_noise_R_Rdomain(R,
                             alpha):

    """
    Args:
        R: An residual-domain image (real or generated).
        alpha: The fraction of R that goes into the output (the remainder is uniform noise).

    Returns:
        if alpha < 1:
            R_noise: alpha * R + (1 - alpha) * noise
            noise = random.uniform(-1,1)
        else:
            R
    """

    R_noisy = tf.cond(tf.less(alpha,
                              1),
                      # if alpha < 1:
                      # return R, noise-ified in the R domain
                      lambda: alpha * R + \
                              (tf.identity(1.0) - alpha) * \
                              K.random_uniform(shape=tf.shape(R),
                                               minval=-1,
                                               maxval=1),
                      # else:
                      # return R with no noise-ification
                      lambda: R)

    return R_noisy

@tf.function
def instance_noise_B_Rdomain(B,
                             A,
                             alpha):

    """
    Args:
        B: An HRI-domin image (real or generated).
        A: The conditioning image, in this case A = up(down(B)).
        alpha: The noise fraction in the R domain.

    Returns:
        if alpha < 1:
            B_noise = alpha * B + (1 - alpha) * (noise + A)
            noise = random.uniform(-1,1)
        else:
            B
    """

    B_noisy = tf.cond(tf.less(alpha,
                              1),
                      # if alpha < 1:
                      # return B, noise-ified in the R domain
                      lambda: alpha * B + \
                              (tf.identity(1.0) - alpha) * \
                              (K.random_uniform(shape=tf.shape(B),
                                                minval=-1,
                                                maxval=1) + A),
                      # else:
                      # return B with no noise-ification
                      lambda: B)

    return B_noisy

###############################################################################

"""
CONVOLUTION BLOCK
"""

def conv(input_tensor,
         n_filters,
         kernel_size,
         strides=(1, 1),
         pool=False,
         bn=False,
         ln=False,
         norm_axis=-1):

    """
    Args:
        input_tensor: the input tensor to the convolutional block (e.g., the output of the previous layer)
        n_filters: the number of filters/kernels in the block
        kernel_size: the size (integer if square, tuple if not) of the kernels
        strides: stride length (integer or tuple)
        pool: Boolean. Whether or not there is a max pooling layer after the convolution.
        bn: Boolean. Whether or not there is a batch normalization layer after the activation function. *Only one of bn or ln should be True.*
        ln: Boolean. Whether or not there is a layer normalization layer after the activation function. *Only one of bn or ln should be True.*
        norm_axis: the axis over which batch normalization or layer normalization is performed.

    Returns:
        The tensor outputs of the following block:
            input
            2D convolution
            [max pool]
            LeakyReLU
            [batch normalization]
            [layer normalization]
            output
        where [...] indicates optional components.
    """

    y = Convolution2D(filters=n_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal')(input_tensor)

    if pool:
        y = MaxPooling2D(pool_size=(2, 2),
                         padding='same')(y)

    y = LeakyReLU()(y)

    if bn:
        y = BatchNormalization(axis=norm_axis)(y)

    if ln:
        y = LayerNormalization(axis=norm_axis)(y)

    return y

"""
TRANSPOSE CONVOLUTION BLOCK
"""

def tconv(input_tensor,
          n_filters,
          kernel_size,
          strides=(1, 1),
          bn=False,
          ln=False,
          norm_axis=-1):

    """
    Args:
        input_tensor: the input tensor to the convolutional block (e.g., the output of the previous layer)
        n_filters: the number of filters/kernels in the block
        kernel_size: the size (integer if square, tuple if not) of the kernels
        strides: stride length (integer or tuple)
        bn: Boolean. Whether or not there is a batch normalization layer after the activation function. *Only one of bn or ln should be True.*
        ln: Boolean. Whether or not there is a layer normalization layer after the activation function. *Only one of bn or ln should be True.*
        norm_axis: the axis over which batch normalization or layer normalization is performed.

    Returns:
        The tensor outputs of the following block:
            input
            2D convolution
            LeakyReLU
            [batch normalization]
            [layer normalization]
            output
        where [...] indicates optional components.
    """

    y = Conv2DTranspose(filters=n_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same')(input_tensor)

    y = LeakyReLU()(y)

    if bn:
        y = BatchNormalization(axis=norm_axis)(y)

    if ln:
        y = LayerNormalization(axis=norm_axis)(y)

    return y

"""
ResNeXt BLOCK
"""

"""
This code was adapted from
https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
with the following changes:
    1) ResNeXt blocks (see https://arxiv.org/pdf/1603.05027.pdf )
    2) batch normalization -> ReLU
       replaced by
       LeakyReLU -> [batch OR layer OR no] normalization
"""

def add_common_layers(y,
                      bn=False,
                      ln=False,
                      norm_axis=-1):

    """
    Args:
        y: input tensor
        bn: Boolean. Whether or not there is a batch normalization layer after the activation function. *Only one of bn or ln should be True.*
        ln: Boolean. Whether or not there is a layer normalization layer after the activation function. *Only one of bn or ln should be True.*
        norm_axis: the axis over which batch normalization or layer normalization is performed.

    Returns:
        The tensor outputs of the following block:
            input
            LeakyReLU
            [batch normalization]
            [layer normalization]
            output

    NOTE: Goes AT THE TOP of the stack to enable us to preserve the identity map. This also means that if this block is not used, activation and normalization must be added manually.
    """

    y = LeakyReLU()(y)

    if bn:
        y = BatchNormalization(axis=norm_axis)(y)
    if ln:
        y = LayerNormalization(axis=norm_axis)(y)

    return y

def grouped_convolution(y,
                        nb_channels,
                        _strides,
                        ksize,
                        cardinality):

    """
    Args:
        y: input tensor
        nb_channels: total number of channels in the block, across all branches
        _strides: strides of the convolution
        ksize: size of the convolution kernels
        cardinality: number of branches in the block

    Returns:
        The tensor outputs of the following stack:
            input
            split into a number of branches equal to cardinality
            2D convolution on each branch
            concatenate the branches back together
            output

    NOTE: nb_channels must be evenly divisible by cardinality (each branch has nb_channels/cardinality filters)
    """

    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Convolution2D(nb_channels,
                             kernel_size=ksize,
                             strides=_strides,
                             padding='same')(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups, and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Convolution2D(_d,
                                    kernel_size=ksize,
                                    strides=_strides,
                                    padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    # CANNOT use Concatenate here. concatenate is the FUNCTIONAL INTERFACE to Concatenate, and that's what we need here.
    y = tf.keras.layers.concatenate(groups)

    return y

def residual_block(y,
                   nb_channels_in,
                   nb_channels_out,
                   _strides=(1, 1),
                   _project_shortcut=False,
                   ksize=(4, 4),
                   cardinality=4,
                   bn=False,
                   ln=False,
                   norm_axis=-1):

    """
    Args:
        y: input tensor
        nb_channels_in: reduced number of channels in the bottleneck/dimension reduction phase
        nb_channels_out: number of channels in the output of the block
        _strides: strides of the convolution
        _project_shortcut: Boolean. ResNeXt blocks have an identity map that is added back into the output of the block; if the number of channels in the output is expected to differ from the number of channels in the input, or if the feature maps change size (e.g., if _strides != 1), the identity path must be transformed. if _project_shortcut, 1x1 convolutions will be performed on the identity path to map it to the appropriate shape.
        ksize: size of the convolution kernels
        cardinality: number of branches in the block
        bn: Boolean. Whether or not there is a batch normalization layer after the activation function. *Only one of bn or ln should be True.*
        ln: Boolean. Whether or not there is a layer normalization layer after the activation function. *Only one of bn or ln should be True.*
        norm_axis: the axis over which batch normalization or layer normalization is performed.

    Returns:
        The tensor outputs of the following stack:
            input
            add_common_layers
            2D convolution (1x1 kernels) <- bottleneck
            add_common_layers
            grouped_convolution
            add_common_layers
            2D convolution (1x1 kernels) <- map to desired number of channels
            add to input (or projected input)
            output

    Two general rules (that we do not always follow for various reasons, including but not limited to parameter budget):
        1) blocks producing spatial maps of the same size as their inputs have the same hyperparameters (cardinality, filter sizes, etc.)
        2) each time a spatial map is downsampled by some factor, the width of the blocks (number of channels) is multiplied by the same factor.
    """

    # This is the identity / shortcut path
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = add_common_layers(y,
                          bn,
                          ln,
                          norm_axis)
    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = add_common_layers(y,
                          bn,
                          ln,
                          norm_axis)

    y = grouped_convolution(y,
                            nb_channels_in,
                            _strides=_strides,
                            ksize=ksize,
                            cardinality=cardinality)

    # Map the aggregated branches to desired number of output channels
    y = add_common_layers(y,
                          bn,
                          ln,
                          norm_axis)

    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same')(y)

    # Add to the shortcut
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut \
    or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Convolution2D(nb_channels_out,
                                 kernel_size=(1, 1),
                                 strides=_strides,
                                 padding='same')(shortcut)

    # Add the shortcut and the transformation block
    # Same as with the concatenate layer in grouped_convolution, we need the FUNCTIONAL INTERFACE with Add, which is add.
    y = tf.keras.layers.add([shortcut, y])

    return y

###############################################################################

"""
FUNCTION MAPS DATA TO RANGE [-1, 1]
"""

def one_one(x):

    shape = x.shape

    x_one_one = 2 * (x - x.min()) / (x.max() - x.min()) - 1

    x_one_one = x_one_one.reshape(shape)

    return x_one_one

###############################################################################

"""
FUNCTION FOR 2x2 AVERAGE POOL DOWNSAMPLING
AND
FUNCTION FOR 2x2 UPSAMPLING
"""

def down(img):

    """
    Args:
        img:  an HxWxD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Returns:
        down(img):  a 2x2 average-pooled downsampling of img -> an (H/2)x(W/2)xD image OR a btach of such images.  In the latter case the inputs have an extra (batch) dimension.

    Intended to be used with map().
    """

    # if performing on individual examples with no batch dimension, need to expand dims to give a "batch size" of 1.
    batch = True

    #if len(tf.shape(img)) == 3:
    if tf.shape(img).shape[0] == 3:
        img = tf.expand_dims(img,
                             axis=0)

        batch = False

    batch_size = tf.shape(img)[0]
    (M, N) = (tf.shape(img)[1], # img height
              tf.shape(img)[2]) # img width
    depth = tf.shape(img)[3]

    (K, L) = (2, 2)

    MK = M // K
    NL = N // L

    downsampled = img[:,
                      :MK*K,
                      :NL*L,
                      :]

    downsampled = tf.reshape(downsampled,
                             shape=(batch_size,
                                    MK, K,
                                    NL, L,
                                    depth))

    downsampled = tf.reduce_mean(downsampled,
                                 axis=(2,
                                       4))

    # if performing on individual examples, need to squeeze out the "batch" dimension.
    if not batch:
        downsampled = tf.squeeze(downsampled,
                                 axis=0)

    return downsampled

def up(img):

    """
    Args:
        img:  an HxWxD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Returns:
        up(img):  a 2x2-repeated upscaling of img -> a (2*H)x(2*W)xD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Intended to be used with map().
    """

    # if performing on individual examples with no batch dimension, need to expand dims to give a "batch size" of 1.
    batch = True

    #if len(tf.shape(img)) == 3:
    if tf.shape(img).shape[0] == 3:
        img = tf.expand_dims(img,
                             axis=0)

        batch = False

    # Upsample in H...
    upscaled = tf.repeat(img,
                         repeats=2,
                         axis=1)
    # ...and W.
    upscaled = tf.repeat(upscaled,
                         repeats=2,
                         axis=2)

    # if performing on individual examples, need to squeeze out the "batch" dimension.
    if not batch:
        upscaled = tf.squeeze(upscaled,
                              axis=0)

    return upscaled

###############################################################################

"""
FUNCTIONS FOR LOADING, PARSING AND PREPROCESSING FASHION-MNIST FROM TFRECORDS FILES TO TENSORFLOW DATASET
"""

def make_dataset(filepath,
                 z_dim,
                 batch_size,
                 shuffle_buffer_size=None):

    """
    Tensorflow input pipeline.

    Args:
        filepath: The filepath to the target tfrecords file, containing both images (x) and one-hot class labels (y). The x data has ALREADY:
            (i) been converted to float,
            (ii) been reshaped appropriately to feed into the network, and
            (iii) been rescaled int [-1, 1] (the training and validation datasets have been rescaled independently from one another).
        z_dim: The dimensionality of the latent space.
        batch_size: The batch size in the resulting dataset.
        shuffle_buffer_size: None if no shuffling is to be performed; otherwise, the size of the shuffle buffer. Fashion-MNIST is small enough that we can shuffle the entire dataset (there are 60000 examples in the training set and 10000 in the validation set).

    Returns:
        dataset: A ready-to-use TensorFlow Dataset (z, up(B2), B1), where
            z: a z_dim-dimensional vector sampled from the standard normal
            B2: 7x7 images created by downsampling fashion-MNIST images TWICE
            B1: 14x14 images created by downsample fashion-MNIST images ONCE
            up(): simple 2x2 upscaling

    ###########################################################################

    Best practices for datasets too large to fit into memory, from https://github.com/tensorflow/tensorflow/issues/14857:
    The Dataset.shuffle() implementation is designed for data that could be shuffled in memory (...) here's the usual approach we use when the data are too large to fit in memory:

    1. Randomly shuffle the entire data once using a MapReduce/Spark/Beam/etc. job to create a set of roughly equal-sized files ("shards").
    2. In each epoch:
        i. Randomly shuffle the list of shard filenames, using Dataset.list_files(...).shuffle(num_shards).
        ii. Use dataset.interleave(lambda filename: tf.data.TextLineDataset(filename), cycle_length=N) to mix together records from N different shards.
        iii. Use dataset.shuffle(B) to shuffle the resulting dataset. Setting B might require some experimentation, but you will probably want to set it to some value larger than the number of records in a single shard.
    """

    def load_fashion_MNIST(filepath):

        """
        Args:
            filepath: The filepath to the target tfrecordsfile.

        Returns:
            serialized_dataset: A Dataset object containing serialized image (x) and one-hot class label (y) data.
        """

        # Create a Dataset from the TFRecord file.
        serialized_dataset = tf.data.TFRecordDataset(filepath)

        return serialized_dataset

    @tf.function
    def parse_example(serialized_example):

        """
        Args:
            serialized_example: A single serialized example from the serialized dataset obtained from load_fashion_MNIST.

        Returns:
            B0: The original 28x28x1 image (with depth channel). Class label data is discarded since we don't need it for now.

        DOES NOT run on a batched dataset.

        Intended to be used with map().
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

        parsed_example = tf.io.parse_single_example(
                                serialized=serialized_example,
                                features=features)

        # Get the image dimensions.
        height = parsed_example['height']
        width = parsed_example['width']
        depth = parsed_example['depth']

        # Get the raw byte string and convert it to a tensor object.
        B0 = parsed_example['img']
        B0 = tf.io.decode_raw(B0,
                              tf.float32)
        # The tensor object is 1-dimensional, so reshape it using the image dimensions we obtained earlier.
        B0 = tf.reshape(B0,
                        shape=(height,
                               width,
                               depth))

        return B0

    @tf.function
    def get_images(B0):

        """
        Args:
            B0: The image outputted by parse_example.

        Returns:
            (B2, B1, B0), with:
                B1 = down(B0)
                B2 = down(B1)
        """

        B1 = down(B0)
        B2 = down(B1)

        B0 = tf.cast(B0, tf.float32)
        B1 = tf.cast(B1, tf.float32)
        B2 = tf.cast(B2, tf.float32)

        return (B2, B1, B0)

    @tf.function
    def get_latent_vectors(B2,
                           B1,
                           B0):

        """
        Args:
            B0: The original image.
            B1 = down(B0)
            B2 = down(B1)

        Returns:
            x = (z2, up(B2), B1)
        """

        z2 = tf.random.normal((z_dim,),
                              dtype=tf.float32)
        return (z2, up(B2), B1)

    dataset = load_fashion_MNIST(filepath)

    dataset = dataset.map(parse_example,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(get_images,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # The B examples will not change from epoch to epoch so they can be cached
    dataset = dataset.cache()

    # The z examples WILL change from epoch to epoch so they must be generated after the cache.
    dataset = dataset.map(get_latent_vectors,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the dataset.
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # Batch the dataset.
    dataset = dataset.batch(batch_size)

    # Prefetch the dataset.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

###############################################################################

"""
KL DIVERGENCE
"""

def KL_batchwise_loss(y_true,
                      y_pred):

    """
    Args:
        y_true: a dummy tensor. tf.keras loss functions must have a y_true, but it is not used to calculate the KL divergence loss
        y_pred: Batch of z vectors with shape (batch_size, z_dim).

    Returns:
        The KL divergence from N(0,1) to the distribution represented by y_pred, evaluated once over the batch. Has a strict lower bound at 0 but no upper bound.

    The KL divergence from a normal distribution p2 to another normal distribution p1 is (http://allisons.org/ll/MML/KL/Normal/ ):

        KL(p1 || p2) = log(sigma_2 / sigma_1) +
                   [(mu_1 - mu_2)^2 + (sigma_1^2 - sigma_2^2)] / 2*sigma_2^2

    When p2 is the standard normal distribution, this simplifies to:

        KL(N(mu, sigma^2) || N(0, 1)) = -log(sigma) +
                                    [mu^2 + sigma^2 - 1] / 2

    NOTE: as in both tensorflow and numpy, log is log_e, the natural logarithm.
    NOTE: this loss function assumes y_pred was sampled from some (non-standard) multivariate normal distribution N(mu,sigma^2).
    """

    mu = K.mean(y_pred,
                axis=0)

    sigma = K.std(y_pred,
                  axis=0)

    KL = -K.log(sigma) + \
         0.5 * (K.square(mu) + \
                K.square(sigma) - \
                1)

    KL = K.mean(KL,
                axis=-1)

    return KL

def KL_variational_loss(y_true,
                        y_pred):

    """
    Args:
        y_true: a dummy tensor. tf.keras loss functions must have a y_true, but it is not used to calculate the KL divergence loss
        y_pred: outputs of variational encoder, with form (z_mu, z_logvar)

    Returns:
        The KL divergence from N(0,1) to N(mu,sigma^2), with mu=z_mu and sigma^2=exp[z_logvar]. Has a strict lower bound at 0 but no upper bound.
    """

    z_mu = y_pred[0]
    z_logvar = y_pred[1]

    KL = 0.5 * (tf.math.square(z_mu) + \
                tf.math.exp(z_logvar) - \
                z_logvar - \
                1)

    KL = K.mean(KL)

    return KL

###############################################################################

"""
REPARAMATERIZATION TRICK

Originally found in:
    D.P. Kingma and M. Welling. Auto-encoding Variational Bayes, ICLR (2014).

The procedure is as follows:
    1) Assume E(B) is a normal distribution (not necessarily standard normal).
    2) Obtain mu and sigma via E(B) = (mu, logvar)
    3) Sample the point cloud represented by E(B) via z ~ mu + sigma * N(0,1). This allows easier backpropagation because the reparameterized sampled z is differentiable with respect to mu and sigma, which are predicted deterministically by E.
    4) Pass this newly sampled z through G to generate a fake image.
"""

def reparameterize(E_of_B):

    """
    Args:
        E_of_B: E(B) = (mu, log[sigma^2])

    Returns:
        z_sampled: mu + sigma * epsilon
        epsilon = N(0,1)
    """

    z_mu = E_of_B[0]
    z_logvar = E_of_B[1]

    batch_size = K.shape(z_mu)[0]
    z_dim = K.shape(z_mu)[1]
    epsilon = K.random_normal(shape=(batch_size,
                                     z_dim))

    z_sigma = K.exp(0.5 * z_logvar)

    z_sampled = z_mu + z_sigma * epsilon

    return z_sampled

###############################################################################

"""
WASSERSTEIN LOSS
"""

"""
Calculates the Wasserstein loss for a sample batch.
    In a Wasserstein GAN, the output is linear (no activation function).  The discriminator tries to make the distance between its prediction for real and fake samples as large as possible.
    This can be done by labeling fake and real samples as -1 and 1, respectively.
    The labels, y_true, have fixed values +/- 1.  However, there is NO restriction on y_pred (and it is just a random scalar for a randomly initialized network).  There is therefore no reason that y_pred should be fixed at +/- 1 or even in [-1, 1], although in practice it shouldn't be too far outside this range.
    With this +/- 1 formulation (instead of the usual range [0, 1] for fake/real), the Wasserstein loss can be very simply calculated by multiplying the outputs and labels.
    Note that as a result of this formulation the loss can and frequently will be negative.
"""

def wasserstein_loss(y_true,
                     y_pred):

    """
    Args:
        y_true: labels (either +1 for real or -1 for fake)
        y_pred: the predicted critic score, more negative for "more likely to be fake," more real for "more likely to be real"

    Returns:
        Wasserstein loss.

    NOTE: The labels, y_true, have fixed values of +/-1. However, there is no similar restriction on y_pred (and in fact, it is a random scalar for a randomly initialized C prior to training). y_pred is not fixed to +/-1 or even restricted to [-1,1], although in practice it should not be too far outside this range.
    NOTE: The +/-1 convention for the score (as oposed to 0/1), the Wasserstein loss is just a simple product. This does mean that the loss can and frequently will be negative.
    """

    return -K.mean(y_true * y_pred)

###############################################################################

@tf.function
def RandomWeightedAverage(tensor_1,
                          tensor_2):

    """
    Args:
        tensor_1: A TensorFlow tensor.
        tensor_2: A TensorFlow tensor with the same shape as tensor_1.

    Returns:
        random_weighted_average: A randomly-weighted average of tensor_1 and tensor_2. (In geometric terms, this outputs a random point on the line between the two input points.)
    """

    shape = K.shape(tensor_1)

    weights = K.random_uniform(shape[:1],
                               0,
                               1)

    for i in range(len(K.int_shape(tensor_1)) - 1):
        weights = K.expand_dims(weights,
                                -1)

    return tensor_1 * weights + tensor_2 * (1 - weights)

###############################################################################

"""
CUSTOM TRAINING LOOP LOSSES (DETERMINISTIC)
"""

def L2_magnitude(y_true,
                 y_pred):

    """
    Args:
        y_true: a dummy tensor. tf.keras loss functions must have a y_true, but it is not used to calculate the L2 magnitude loss
        y_pred: the tensor whose L2 magnitude is to be calculated

    Returns:
        ||y_pred||_2

    NOTE: This loss function is used for the consistency loss, since for 2x2 average pooling corruption, the consistency loss can be explicitly written as ||down(R)||_2, where down(...) is 2x2 average pool and R is the residual.
    """

    return K.sqrt(K.sum(K.square(y_pred)))

def generator_loss(R_real,
                   R_generated,
                   R_cycled,
                   critic_verdict_R_generated,
                   critic_verdict_R_cycled,
                   CONSISTENT,
                   loss_weights):

    """
    Args:
        R_real: The ground truth residual
            R = B - A
            with A = up(down(B)) in this case.
        R_generated: The residual generated from z_real,
            R_gen = G(z_real,
                      A).
        R_cycled: The residual obtained from
            R_cyc = G(z_enc,
                      A)
            with z_enc = E(R,A) (deterministic)
            OR z_enc = reparameterize[E(R,A)] (variational)
        critic_verdict_R_generated: The critic's verdict on R_gen.
        critic_verdict_R_cycled: The critic's verdict on R_cyc.
        CONSISTENT: Boolean. Whether or not to include self-consistency terms in the generator loss function.
        loss_weights: An array of loss weights with which to weight the terms in the generator loss function.
            loss_weights = [LAMBDA_R_RECONSTRUCTION,
                            LAMBDA_CRITIC_cLR,
                            LAMBDA_CRITIC_cAE,
                            LAMBDA_CONSISTENCY_cLR,
                            LAMBDA_CONSISTENCY_cAE]

    Returns:
        total_loss: The total generator loss (a sum of everything below).
        R_reconstruction_loss:
            LAMBDA_R_RECONSTRUCTION * ||R - R_cyc||_1
        wasserstein_loss_cLR:
            LAMBDA_CRITIC_cLR * -(1 *
                                  critic_verdict_R_generated)
        wasserstein_loss_cAE:
            LAMBDA_CRITIC_cAE * -(1 *
                                  critic_verdict_R_cycled)
        if CONSISTENT:
            consistency_loss_cLR:
                LAMBDA_CONSISTENCY_cLR * ||down(R_generated)||_2
            consistency_loss_cAE:
                LAMBDA_CONSISTENCY_cAE * ||down(R_cycled)||_2

    Note on the consistency terms:
        In general, the consistency loss looks like
            consistency = ||corruption(B') - corruption(B)||_2.
        In our case, corruption(X) = up(down(X)).
        Recall A = corruption(B).
        Then
            consistency = ||up(down(B')) - A||_2
        Since
            down(X + Y) = down(X) + down(Y)
            up(X + Y) = up(X) + up(Y)
            down(up(X)) = X
        and
            B' = R' + A
            R = B - A
        we have
            consistency = ||up(down(R' + A)) - A||_2
                        = ||up(down(R') + down(A)) - A||_2
                        = ||up(down(R')) + up(down(A)) - A||_2
                        = ||up(down(R'))||_2
                        = 4*||down(R')||_2
        During the last step, recall that up(X) is just a 2x2 upscaling of X.
        Thus, the consistency loss can be simply rewritten as a function of the downsampled residual only. (Equivalent reductions may not exist for all corruption types, or for other definitions of "consistency.") In this case, minimizing the consistency loss is equivalent to ensuring that the pixels of the residual in each 2x2 (stride 2) patch sum to zero.
    We use the L2 norm in the consistency term because we are more concerned about penalizing outliers (patches where the sum is far from zero) than encouraging sparsity.
    """

    # The individual loss weights are:
    LAMBDA_R_RECONSTRUCTION_G = loss_weights[0]
    LAMBDA_CRITIC_cLR_G = loss_weights[1]
    LAMBDA_CRITIC_cAE_G = loss_weights[2]
    LAMBDA_CONSISTENCY_cLR_G = loss_weights[3]
    LAMBDA_CONSISTENCY_cAE_G = loss_weights[4]

    # Define a helper function for the MAE loss
    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()

    # 1st term
    R_reconstruction_loss = mean_absolute_error(R_real,
                                                R_cycled)
    R_reconstruction_loss *= LAMBDA_R_RECONSTRUCTION_G

    # 2nd term
    wasserstein_loss_cLR = wasserstein_loss(
                                tf.ones_like(critic_verdict_R_generated),
                                critic_verdict_R_generated)
    wasserstein_loss_cLR *= LAMBDA_CRITIC_cLR_G

    # 3rd term
    wasserstein_loss_cAE = wasserstein_loss(
                                tf.ones_like(critic_verdict_R_cycled),
                                critic_verdict_R_cycled)
    wasserstein_loss_cAE *= LAMBDA_CRITIC_cAE_G

    if CONSISTENT:

        # 4th term
        R_gen_down = down(R_generated)
        consistency_loss_cLR = L2_magnitude(R_gen_down,
                                            R_gen_down)
        consistency_loss_cLR *= LAMBDA_CONSISTENCY_cLR_G

        # 5th term
        R_cyc_down = down(R_cycled)
        consistency_loss_cAE = L2_magnitude(R_cyc_down,
                                            R_cyc_down)
        consistency_loss_cAE *= LAMBDA_CONSISTENCY_cAE_G

    # Compute the total loss
    if not CONSISTENT:
        total_loss = R_reconstruction_loss + \
                     wasserstein_loss_cLR + \
                     wasserstein_loss_cAE

    else:
        total_loss = R_reconstruction_loss + \
                     wasserstein_loss_cLR + \
                     wasserstein_loss_cAE + \
                     consistency_loss_cLR + \
                     consistency_loss_cAE

    # Return the total as well as the individual terms
    if not CONSISTENT:
        return (total_loss,
                R_reconstruction_loss,
                wasserstein_loss_cLR,
                wasserstein_loss_cAE)
    else:
        return (total_loss,
                R_reconstruction_loss,
                wasserstein_loss_cLR,
                wasserstein_loss_cAE,
                consistency_loss_cLR,
                consistency_loss_cAE)

def encoder_loss(z_real,
                 z_cycled,
                 z_encoded,
                 loss_weights):

    """
    For the DETERMINISTIC version.

    Args:
        z_real: A latent space vector sampled from the target distribution,
            z_real ~ p(z) = N(0,1).
        z_cycled: The latent space vector z_cyc = E(R_gen,A),
            with R_gen = G(z,A).
        z_encoded: The latent space vector z_enc = E(R,A).

        loss_weights: An array of loss weights with which to weight the terms in the encoder loss function.
            loss_weights = [LAMBDA_Z_RECONSTRUCTION,
                            LAMBDA_KL_cLR,
                            LAMBDA_KL_cAE]

    Returns:
        total_loss: The total encoder loss (a sum of everything below).
        z_reconstruction_loss:
            LAMBDA_RECONSTRUCT_Z * ||z_real - z_cycled||_1
        KL_cLR:
            KL_cLR = LAMBDA_KL_cLR * KL[z_cycled' || N(0,
                                                       1)]
        KL_cAE:
            KL_cAE = LAMBDA_KL_cAE * KL[z_encoded' || N(0,
                                                        1)]
            where KL[p || q] is the KL-divergence from q to p and z_cycled' and z_encoded' are batches of z_cycled and z_encoded, respectively.
    """

    # The individual loss weights are:
    LAMBDA_Z_RECONSTRUCTION_E = loss_weights[0]
    LAMBDA_KL_cLR = loss_weights[1]
    LAMBDA_KL_cAE = loss_weights[2]

    # Define a helper function for the MAE loss
    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()

    # 1st term
    z_reconstruction_loss = mean_absolute_error(z_real,
                                                z_cycled)
    z_reconstruction_loss *= LAMBDA_Z_RECONSTRUCTION_E

    # 2nd term
    KL_cLR = KL_batchwise_loss(z_cycled,
                               z_cycled)

    KL_cLR *= LAMBDA_KL_cLR

    # 3rd term
    KL_cAE = KL_batchwise_loss(z_encoded,
                               z_encoded)

    KL_cAE *= LAMBDA_KL_cAE

    # Compute the total loss
    total_loss = z_reconstruction_loss + \
                 KL_cLR + \
                 KL_cAE

    # Return the total as well as the individual terms
    return (total_loss,
            z_reconstruction_loss,
            KL_cLR,
            KL_cAE)

def encoder_loss_var(z_real,
                     z_cycled,
                     z_encoded,
                     loss_weights):

    """
    For the VARIATIONAL version.

    Args:
        z_real: A latent space vector sampled from the target distribution,
            z_real ~ p(z) = N(0,1).
        z_cycled: The tuple (mu,logvar) = E(R_gen,A),
            with R_gen = G(z,A).
        z_encoded: The tuple (mu,logvar) = E(R,A).

        loss_weights: An array of loss weights with which to weight the terms in the encoder loss function.
            loss_weights = [LAMBDA_Z_RECONSTRUCTION,
                            LAMBDA_KL_cLR,
                            LAMBDA_KL_cAE]

    Returns:
        total_loss: The total encoder loss (a sum of everything below).
        z_reconstruction_loss:
            LAMBDA_RECONSTRUCT_Z * ||z_real - z_cycled||_1
        KL_cLR:
            KL_cLR = LAMBDA_KL_cLR * KL[z_cycled || N(0,
                                                      1)]
        KL_cAE:
            KL_cAE = LAMBDA_KL_cAE * KL[z_encoded || N(0,
                                                       1)]
            where KL[p || q] is the KL-divergence from q to p calculated using the mu and logvar outputs of E.
    """

    # The individual loss weights are:
    LAMBDA_Z_RECONSTRUCTION_E = loss_weights[0]
    LAMBDA_KL_cLR = loss_weights[1]
    LAMBDA_KL_cAE = loss_weights[2]

    # Define a helper function for the MAE loss
    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()

    # 1st term
    z_reconstruction_loss = mean_absolute_error(z_real,
                                                z_cycled[0])
    z_reconstruction_loss *= LAMBDA_Z_RECONSTRUCTION_E

    # 2nd term
    KL_cLR = KL_variational_loss(z_cycled,
                                 z_cycled)

    KL_cLR *= LAMBDA_KL_cLR

    # 3rd term
    KL_cAE = KL_variational_loss(z_encoded,
                                 z_encoded)

    KL_cAE *= LAMBDA_KL_cAE

    # Compute the total loss
    total_loss = z_reconstruction_loss + \
                 KL_cLR + \
                 KL_cAE

    # Return the total as well as the individual terms
    return (total_loss,
            z_reconstruction_loss,
            KL_cLR,
            KL_cAE)

def critic_loss(critic_verdict_R_real,
                critic_verdict_R_generated,
                critic_verdict_R_cycled,
                random_weighted_average_cLR,
                critic_verdict_averaged_cLR,
                random_weighted_average_cAE,
                critic_verdict_averaged_cAE,
                loss_weights):

    """
    Args:
        critic_verdict_R_real: The critic's verdict on R_real.
        critic_verdict_R_generated: The critic's verdict on R_generated.
        critic_verdict_R_cycled: The critic's verdict on R_cycled.
        random_weighted_average_cLR: A randomly weighted average of R_real and R_generated.
        critic_verdict_averaged_cLR: The critic's verdict on random_weighted_average_cLR.
        random_weighted_average_cAE: A randomly weighted average of R_real and R_cycled.
        critic_verdict_averaged_cAE: The critic's verdict on random_weighted_average_cAE.
        loss_weights: An array of loss weights with which to weight the terms in the critic loss function.
            loss_weights = [LAMBDA_CRITIC,
                            LAMBDA_GP]

    Returns:
        total_loss: The total critic loss (a sum of everything below).
        wasserstein_loss_real:
            LAMBDA_CRITIC * -(1 *
                              critic_verdict_R_real)
        wasserstein_loss_cLR:
            (LAMBDA_CRITIC / 2) * -((-1) *
                                    critic_verdict_R_generated)
        wasserstein_loss_cAE:
            (LAMBDA_CRITIC / 2) * -((-1) *
                                    critic_verdict_R_cycled)
        gradient_penalty_cLR:
            LAMBDA_GP * gradient_penalty(random_weighted_average_cLR,
                                         critic_verdict_averaged_cLR)
        gradient_penalty_cAE:
            LAMBDA_GP * gradient_penalty(random_weighted_average_cAE,
                                         critic_verdict_averaged_cAE)

        The gradient penalty is defined by:
            gradients = the gradients of critic_verdict_averaged with respect to random_weighted_average
            gradient_penalty = (1 - ||gradients||_2)^2
        The verdict has dimensions (batch_size, 1) and random_weighted_average has dimensions (batch_size, number_of_features). The gradient penalty then has dimensions (batch_size, number_of_features) as well.
        To enforce the 1-Lipschitz constraint (the gradient norm has upper bound 1), the loss function penalizes the network if the gradient norm with respect to the input averaged samples (as opposed to the gradient norm w.r.t. the weights of the discriminator) exceeds 1.
        It is impossible to evaluate this function at all points in the input space.  The compromise is to choose random points on the lines between real and generated samples and spot check the gradients at these points.  The assumption is that these spot checks are a reasonable guarantee of 1-Lipschitz-ness.
    """

    # The individual loss weights are:
    LAMBDA_CRITIC = loss_weights[0]
    LAMBDA_GP = loss_weights[1]

    # 1st term
    wasserstein_loss_real = wasserstein_loss(
                                tf.ones_like(critic_verdict_R_real),
                                critic_verdict_R_real)
    wasserstein_loss_real *= LAMBDA_CRITIC

    # 2nd term
    wasserstein_loss_cLR = wasserstein_loss(
                                -tf.ones_like(critic_verdict_R_generated),
                                critic_verdict_R_generated)
    wasserstein_loss_cLR *= (0.5 * LAMBDA_CRITIC)

    # 3rd term
    wasserstein_loss_cAE = wasserstein_loss(
                                -tf.ones_like(critic_verdict_R_cycled),
                                critic_verdict_R_cycled)
    wasserstein_loss_cAE *= (0.5 * LAMBDA_CRITIC)

    # 4th term
    gradient_penalty_cLR = K.gradients(critic_verdict_averaged_cLR,
                                       random_weighted_average_cLR)[0]
    gradient_penalty_cLR = K.square(gradient_penalty_cLR)
    gradient_penalty_cLR = K.batch_flatten(gradient_penalty_cLR)
    gradient_penalty_cLR = K.sum(gradient_penalty_cLR,
                                 axis=1,
                                 keepdims=True)
    gradient_penalty_cLR = K.sqrt(gradient_penalty_cLR)
    gradient_penalty_cLR = 1 - gradient_penalty_cLR
    gradient_penalty_cLR = K.square(gradient_penalty_cLR)
    gradient_penalty_cLR = K.mean(gradient_penalty_cLR)
    gradient_penalty_cLR *= LAMBDA_GP

    # 5th term
    gradient_penalty_cAE = K.gradients(critic_verdict_averaged_cAE,
                                       random_weighted_average_cAE)[0]
    gradient_penalty_cAE = K.square(gradient_penalty_cAE)
    gradient_penalty_cAE = K.batch_flatten(gradient_penalty_cAE)
    gradient_penalty_cAE = K.sum(gradient_penalty_cAE,
                                 axis=1,
                                 keepdims=True)
    gradient_penalty_cAE = K.sqrt(gradient_penalty_cAE)
    gradient_penalty_cAE = 1 - gradient_penalty_cAE
    gradient_penalty_cAE = K.square(gradient_penalty_cAE)
    gradient_penalty_cAE = K.mean(gradient_penalty_cAE)
    gradient_penalty_cAE *= LAMBDA_GP

    # Compute the total loss
    total_loss = wasserstein_loss_real + \
                 wasserstein_loss_cLR + \
                 wasserstein_loss_cAE + \
                 gradient_penalty_cLR + \
                 gradient_penalty_cAE

    # Return the total loss as well as the individual terms
    return (total_loss,
            wasserstein_loss_real,
            wasserstein_loss_cLR,
            wasserstein_loss_cAE,
            gradient_penalty_cLR,
            gradient_penalty_cAE)