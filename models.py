#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:56:25 2020

@author: student
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LayerNormalization,
                                     LeakyReLU,
                                     Dense,
                                     Flatten,
                                     Input,
                                     Reshape,
                                     Convolution2D,
                                     RepeatVector,
                                     Concatenate,
                                     Add)

import sys
sys.path.append('.')
from base_functions import residual_block

###############################################################################

"""
LIST OF FUNCTIONS

make_generator_model
make_encoder_model
make_critic_model
"""

###############################################################################

"""
BUILD CONDITIONED SR-GAN TO RECONSTRUCT ORIGINAL IMAGE FROM DECIMATED IMAGE USING ResNeXt BLOCKS
"""

def make_generator_model(z_dim,
                         x_shape,
                         LAYER_NORM,
                         TESTING,
                         pretrained_model_filepath=None):

    """
    Args:
        z_dim: The dimensionality of the latent space that maps to the space of reconstructions produced by G.
        x_shape: The shape of the target reconstruction (same as the shape of the inputs, since they are upsized once before being used as conditioners).
        LAYER_NORM: Boolean. Whether or not to use LayerNormalization in the defined models.
        TESTING: Boolean. If true, removes DEPTH blocks that do not change the shape of the feature space; this reduces the number of parameters in the models and speeds up testing. Set TESTING = True if you want to verify that the overall training scheme works with a toy version of your model.
        pretrained_model_filepath: Either None or a string. If a string, the string is a filepath, and make_generator_model returns the generator model located at that filepath.

    Returns:
        A generator model, G.

        G has two inputs:
            (i) A z_dim-dimensional vector, z, in the latent space, and
            (ii) A conditioning input, A.
        G has one output:
            (I) A residual image, R = B - A, where B is the reconstruction.
    """

    if pretrained_model_filepath:
        G = tf.keras.models.load_model(pretrained_model_filepath)
        return G

    else:

        x_height = x_shape[0] # height of target reconstruction
        x_width = x_shape[1] # width of target reconstruction
        x_depth = x_shape[2] # depth of target reconstruction

        # (i) z
        z_branch_input = Input(shape=(z_dim,),
                               name='z_branch_input_G')

        z_branch = RepeatVector(x_height * x_width)(z_branch_input)

        if K.image_data_format() == 'channels_last':
            z_branch = Reshape((x_height,
                                x_width,
                                z_dim))(z_branch)

            conc_axis = -1

        ### BLOCK z1 ######################################################
        # Transforming the z input by itself ##############################
        # and increasing the number of channels to 256. ###################
        ###################################################################
        z_branch = Convolution2D(filters=256,
                            kernel_size=4,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(z_branch)

        ###################################################################
        ###################################################################
        ###################################################################

        # (ii) A
        c_branch_input = Input(shape=x_shape,
                               name='c_branch_input_G')

        ### BLOCK c1 ######################################################
        # Transforming the conditioner input by itself ####################
        # and increasing the number of channels to 256. ###################
        ###################################################################
        c_branch = Convolution2D(filters=256,
                            kernel_size=4,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(c_branch_input)

        ###################################################################
        ###################################################################
        ###################################################################

        # Concatenate the two branches (512 channels total).
        model = Concatenate(axis=conc_axis)([z_branch,
                                             c_branch])

        if not TESTING:

            ### BLOCK 1 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 2 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 3 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 4 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   ln=LAYER_NORM)

            ### BLOCK 5 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   256,
                                   cardinality=16,
                                   ln=LAYER_NORM)

            ### BLOCK 6 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   256,
                                   cardinality=16,
                                   ln=LAYER_NORM)

            ### BLOCK 7 ###################################################
            # Depth (128 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   128,
                                   _project_shortcut=True,
                                   cardinality=8,
                                   ln=LAYER_NORM)

            ### BLOCK 8 ###################################################
            # Depth (64 channels). ########################################
            ###############################################################
            model = residual_block(model,
                                   128,
                                   64,
                                   _project_shortcut=True,
                                   cardinality=4,
                                   ln=LAYER_NORM)

            ### BLOCK 9 ###################################################
            # Depth (32 channels). ########################################
            ###############################################################
            model = residual_block(model,
                                   64,
                                   32,
                                   _project_shortcut=True,
                                   cardinality=2,
                                   ln=LAYER_NORM)

        else:

            # Simplified test block:
            model = residual_block(model,
                                   512,
                                   32,
                                   _project_shortcut=True,
                                   cardinality=32,
                                   ln=LAYER_NORM)

        ###################################################################
        # Final output layer uses tanh activation (output is a residual).
        # Activation and normalization is placed at the beginning of each residual block, not the end, so they must be added manually here.
        ###################################################################
        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)

        # The fake residual is then:
        model = Convolution2D(x_depth,
                              (3, 3),
                              padding='same',
                              activation='tanh')(model)

        # Define output shape to avoid 'None' shape errors later.
        fake_residual = Reshape(x_shape)(model)

        G = Model([z_branch_input,
                   c_branch_input],
                  fake_residual,
                  name='generator')

        return G

def make_encoder_model(z_dim,
                       x_shape,
                       DETERMINISTIC,
                       LAYER_NORM,
                       TESTING,
                       pretrained_model_filepath=None):

    """
    Args:
        z_dim: The dimensionality of the latent space that E maps its input(s) into.
        x_shape: The shape of the input(s).
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        LAYER_NORM: Boolean. Whether or not to use LayerNormalization in the defined models.
        TESTING: Boolean. If true, removes DEPTH blocks that do not change the shape of the feature space; this reduces the number of parameters in the models and speeds up testing. Set TESTING = True if you want to verify that the overall training scheme works with a toy version of your model.
        pretrained_model_filepath: Either None or a string. If a string, the string is a filepath, and make_encoder_model returns the encoder model located at that filepath.

    Returns:
        An encoder model, E.

        E has two inputs:
            (i) The residual, R = B - up(A)
            (ii) A
        if DETERMINISTIC:
            E has one output:
                (I) A single point in z-space.
        else:
            E has two outputs:
                (I) mu, the center of a point cloud.
                (II) logvar, the log variance of that point cloud.
    """

    if pretrained_model_filepath:
        E = tf.keras.models.load_model(pretrained_model_filepath)
        return E

    else:

        # (i)
        r_branch_input = Input(shape=x_shape,
                               name='r_branch_input')

        if K.image_data_format() == 'channels_last':
            conc_axis = -1

        ### BLOCK i ###################################################
        # Move to a space with a large number of channels (512). ######
        ###############################################################
        r_branch = Convolution2D(filters=512,
                        kernel_size=4,
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(r_branch_input)

        # (ii)
        c_branch_input = Input(shape=x_shape,
                               name='c_branch_input')

        ### BLOCK i1 ##################################################
        # Move to a space with a large number of channels (512). ######
        ###############################################################
        c_branch = Convolution2D(filters=512,
                        kernel_size=4,
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(c_branch_input)


        # Could use concatenate here; using add makes a bit more sense from a conceptual perspective for the encoder specifically.
        model = Add()([r_branch,
                       c_branch])

        if not TESTING:

            ### BLOCK 1 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=64,
                                   ln=LAYER_NORM)

            ### BLOCK 2 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 3 ###################################################
            # Depth (512 channels). #######################################
            # Also, downsample by a factor of 2. ##########################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   _strides=(2,2),
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 4 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   ln=LAYER_NORM)

            ### BLOCK 6 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   128,
                                   _project_shortcut=True,
                                   cardinality=8,
                                   ln=LAYER_NORM)

            ### BLOCK 7 ###################################################
            # Depth (32 channels). ########################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   32,
                                   _project_shortcut=True,
                                   cardinality=2,
                                   ln=LAYER_NORM)

        else:

            # Simplified test block:
            model = residual_block(model,
                                   512,
                                   32,
                                   _strides=(2,2),
                                   _project_shortcut=True,
                                   cardinality=32,
                                   ln=LAYER_NORM)

        ### DENSE BLOCK ###############################################
        # Flatten and reduce to a z output. ###########################
        # Manually add pre-activations. ###############################
        ###############################################################
        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)

        model = Flatten()(model) # M*M*(n_filters in prev. layer) elements
                                 # M is half the size of R
        model = Dense(2*z_dim)(model) # 2*z_dim elements

        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)

        if DETERMINISTIC:
            # This is a deterministic model, so output z
            z = Dense(z_dim,
                      activation='linear',
                      name='z')(model)

            E = Model([r_branch_input,
                       c_branch_input],
                      [z],
                      name='encoder')

        else:
            # This is a variational model, so output (mu, logvar)
            z_mu = Dense(z_dim,
                         activation='linear',
                         name='z_mu')(model)
            z_logvar = Dense(z_dim,
                             activation='linear',
                             name='z_logvar')(model)
            E = Model([r_branch_input,
                       c_branch_input],
                      [z_mu,
                       z_logvar],
                      name='encoder')

        return E

def make_critic_model(x_shape,
                      critic_input,
                      LAYER_NORM,
                      TESTING,
                      pretrained_model_filepath=None):

    """
    Args:
        x_shape: The shape of the input image C will be asked to render verdicts on.
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        LAYER_NORM: Boolean. Whether or not to use LayerNormalization in the defined models.
        TESTING: Boolean. If true, removes DEPTH blocks that do not change the shape of the feature space; this reduces the number of parameters in the models and speeds up testing. Set TESTING = True if you want to verify that the overall training scheme works with a toy version of your model.
        pretrained_model_filepath: Either None or a string. If a string, the string is a filepath, and make_generator_model returns the critic model located at that filepath.

    Returns:
        A critic model, C.

        if critic_input=='AB':

            C has two inputs:
                (i) A, and
                (ii) B
            C has one output:
                (I) The likelihood that B is the original image corresponding to A, or that B was generated by G from A.

        elif critic_input=='B':

            C has one input:
                (i) B
            C has one output:
                (I) The likelihood that B is real, or that B was generated by G.

        elif critic_input=='R':

            C has one input:
                (i) The residual R
            C has one output:
                (I) The likelihood that R is real, or that R was generated by G.

    ###########################################################################

    NOTE: batch normalization is never used in a Wasserstein critic because it creates a correlation between samples in the same batch, which decreases the effectiveness of the gradient penalty loss in enforcing 1-Lipschitz-ness. Layer normalization does not have this problem and is implemented here.
    """

    if pretrained_model_filepath:
        C = tf.keras.models.load_model(pretrained_model_filepath)
        return C

    else:

        # (i)
        i_branch_input = Input(shape=x_shape,
                               name='i_branch_input')

        if K.image_data_format() == 'channels_last':
            conc_axis = -1

        ###############################################################
        # Move to a space with a large number of channels (256 if critic_input = 'AB', 512 otherwise).
        ###############################################################
        if critic_input=='AB':
            i_branch = Convolution2D(filters=256,
                            kernel_size=4,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(i_branch_input)
        else:
            i_branch = Convolution2D(filters=512,
                            kernel_size=4,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(i_branch_input)

        # (ii)
        if critic_input=='AB':

            c_branch_input = Input(shape=x_shape,
                                   name='c_branch_input')

            ###############################################################
            # Move to a space with a large number of channels (256).
            ###############################################################
            c_branch = Convolution2D(filters=256,
                            kernel_size=4,
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(c_branch_input)

        # if critic_input=='AB':  concatenate i_branch and c_branch
        # else:  just use i_branch (c_branch does not exist)
        if critic_input=='AB':
            model = Concatenate(axis=conc_axis)([i_branch,
                                                 c_branch])
        else:
            model = i_branch

        if not TESTING:

            ### BLOCK 1 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 2 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 3 ###################################################
            # Depth (512 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   512,
                                   cardinality=32,
                                   ln=LAYER_NORM)

            ### BLOCK 4 ###################################################
            # Depth (256 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   512,
                                   256,
                                   _project_shortcut=True,
                                   cardinality=16,
                                   ln=LAYER_NORM)

            ### BLOCK 5 ###################################################
            # Depth (128 channels). #######################################
            ###############################################################
            model = residual_block(model,
                                   256,
                                   128,
                                   _project_shortcut=True,
                                   cardinality=8,
                                   ln=LAYER_NORM)

            ### BLOCK 6 ###################################################
            # Keep the number of parameters in the dense blocks from exploding.
            ###############################################################
            model = residual_block(model,
                                   128,
                                   8,
                                   _project_shortcut=True,
                                   cardinality=1,
                                   ln=LAYER_NORM)

        else:

            # Ssimplified test block:
            model = residual_block(model,
                                   512,
                                   8,
                                   _project_shortcut=True,
                                   cardinality=8,
                                   ln=LAYER_NORM)

        ### DENSE BLOCKS ##############################################
        # Flatten and reduce to scalar critic output. #################
        # Manually add pre-activations. ###############################
        ###############################################################

        # Don't forget to pre-activate for the dense layers...
        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)

        model = Flatten()(model)

        model = Dense(96,
                      kernel_initializer='he_normal')(model)

        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)
        model = Dense(24,
                      kernel_initializer='he_normal')(model)

        model = LeakyReLU()(model)
        if LAYER_NORM:
            model = LayerNormalization()(model)
        verdict = Dense(1,
                        kernel_initializer='he_normal')(model)

        if critic_input=='AB':
            C = Model([i_branch_input,
                       c_branch_input],
                      verdict,
                      name='critic')
        else:
            C = Model([i_branch_input],
                      verdict,
                      name='critic')

        return C