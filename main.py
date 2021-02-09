#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:56:37 2020

@author: student
"""

###############################################################################

"""
MODULES
"""

import tensorflow as tf

#from tensorflow.keras.utils import plot_model

import numpy as np

#from scipy.stats import median_absolute_deviation

import time
import os
import sys
sys.path.append('.')
sys.path.append('<<<FOB DIR GOES HERE>>>')
from base_functions import (make_dataset,
                            up,
                            down,
                            generator_loss,
                            encoder_loss,
                            critic_loss,
                            RandomWeightedAverage,
                            instance_noise_alpha,
                            instance_noise_B_Bdomain,
                            instance_noise_B_Rdomain,
                            instance_noise_R_Bdomain,
                            instance_noise_R_Rdomain,
                            reparameterize)

from models import (make_generator_model,
                    make_encoder_model,
                    make_critic_model)

###############################################################################

"""
PARAMETERS
"""

# Build a simplified version of the models (fewer depth blocks) to debug code?
TESTING = True

# Are you using a deterministic encoder?
# If you are using a variational encoder, this is False.
DETERMINISTIC = True

# How many BATCHES without improvement in critic validation loss to consecutively train C per update of G/E?
if TESTING:
    critic_patience = 2
else:
    critic_patience = 5

# Number of validation batches to evaluate whether or not a critic has converged per model update.
if TESTING:
    num_val_batches = 1
else:
    num_val_batches = 5

# How many EPOCHS without improvement in ANY of generator, encoder, and critic validation losses to continue train the models?
# Note that this should not start incrementing til instance noise is done annealing, as some loss functions do not apply in that regime and return absurdly low values.
if TESTING:
    training_patience = 4
else:
    training_patience = 20

# What is the critic input?
# 'B':  C is a function of B only, C(B).
# 'R':  C is a function of R only, C(R).
# 'AB':  C is a function of B, conditioned on A, C(A,B).
critic_input = 'R'
# Need to label output files correctly.
critic_input_label = f'CriticInput{critic_input}'

# Are you using an explicit consistency loss?
CONSISTENT = True
# Need to label output files correctly.
if CONSISTENT:
    consistency_label = 'consistent'
else:
    consistency_label = 'NOTconsistent'

# What dimension of latent space?
z_dim = 100

# Save the models and losses every <this many> epochs.  If it is <= 0, do not save.
if TESTING:
    SAVE_INTERVAL = 0
else:
    SAVE_INTERVAL = 1

# Where are the tfrecords files stored?
# This is also where the demonstration data is stored.
tfrecords_filepath = '/home/student/Work/Keras/GAN/mySRGAN/new_implementation/multiGPU_datasetAPI/custom_train_loop_function_method/DATA_FOR_AAAI/code/'

# What are the shapes of the input/output images?
x_shape = (14, 14, 1)

# If you are loading pretrained files, where are they located?
# If not, these should be None
pretrained_G_filepath = None
pretrained_E_filepath = None
pretrained_C_filepath = None

# Are you restoring from a previous checkpoint (e.g., if you are continuing training after running out of wall time)?
# If not, this should be None. Otherwise, it should be '.../ckpt-#', where # is the desired checkpoint to be restored.
ckpt_path = None

# How many epochs to train?
# This is a maximum value, and if the patience condition is met early, training will stip before reaching this.
# begin_epoch can be set to maintain consistency in file naming if training had to be restarted for whatever reason.
begin_epoch = 0
if TESTING:
    begin_epoch = 0
    fin_epoch = 2
else:
    fin_epoch = 10000

# What is the cutoff epoch for fading out instance noise? alpha will anneal from 0 to 1 over this many epochs.
if TESTING:
    noisy_epochs = 1
else:
    noisy_epochs = 99

# Number of training data per model update.
# Should be a divisor of 60,000 (the total number of training examples in fashion-MNIST).
batch_size = 200

# Are you doing layer normalization or not?
LAYER_NORM = True

###############################################################################

"""
LAMBDAS
"""

"""
Weights to multiply each loss term by when calculating the total loss.

###############################################################################
###############################################################################
###############################################################################

G LOSSES

if not CONSISTENT:
    [total,
     ||R_cyc - R||_1,
     -(1*C(R_gen)) (cLR),
     -(1*C(R_cyc)) (cAE)]

else:
    [total,
     ||R_cyc - R||_1,
     -(1*C(R_gen)) (cLR),
     -(1*C(R_cyc)) (cAE),
     ||down(R_gen)||_2 (cLR),
     ||down(R_cyc)||_2 (cAE)]

###############################################################################

E LOSSES

if DETERMINISTIC:
    [total,
     ||z - z_cyc||_1,
     KL[z_cyc, N(0,1)] (cLR), <- EVALUATED OVER ENTIRE BATCH
     KL[z_enc, N(0,1)] (cAE)] <- EVALUATED OVER ENTIRE BATCH

else:
    [total,
     ||z - mu||_1,
     KL[N(mu_cyc,var), N(0,1)],
     KL[N(mu_enc,var), N(0,1)]]

###############################################################################

C LOSSES

[total,
 -(1*C(R)),
 -(-1*C(R_gen))/2 (cLR),
 -(-1*C(R_cyc))/2 (cAE),
 GP (cLR),
 GP (cAE)]
"""

# Multiply the gradient penalty by this number (to increase it to the point that it can compete with Wasserstein loss).  The original paper used 10. This value is used for the GP loss when updating C.
LAMBDA_GP = 10

# How much weight to accord the critic verdict (cLR and cAE paths) when updating G?
LAMBDA_CRITIC_cLR = 1
LAMBDA_CRITIC_cAE = 1

# In C? This value is used for the GP loss when updating C.
LAMBDA_CRITIC_C = 1

# How much weight to accord the batch-wise KL divergence (deterministic) or point-cloud-based KL divergence (variational) when updating E?
LAMBDA_KL_cLR = 1
LAMBDA_KL_cAE = 1

# How much weight to accord the downsampling consistency term when updating G?
LAMBDA_CONSISTENCY_cLR = 0.1
LAMBDA_CONSISTENCY_cAE = 0.1

# How much weight to accord the reconstruction terms in G and E, respectively??
LAMBDA_R_RECONSTRUCTION = 1
LAMBDA_Z_RECONSTRUCTION = 10

# The loss weight lists are then:
G_loss_weights = [LAMBDA_R_RECONSTRUCTION,
                  LAMBDA_CRITIC_cLR,
                  LAMBDA_CRITIC_cAE,
                  LAMBDA_CONSISTENCY_cLR,
                  LAMBDA_CONSISTENCY_cAE]

E_loss_weights = [LAMBDA_Z_RECONSTRUCTION,
                  LAMBDA_KL_cLR,
                  LAMBDA_KL_cAE]

C_loss_weights = [LAMBDA_CRITIC_C,
                  LAMBDA_GP]

###############################################################################

"""
LOAD & PRE-PROCESS THE DATASET
"""

x_train = tfrecords_filepath + 'xy_train_fMNIST.tfrecords'
x_val = tfrecords_filepath + 'xy_val_fMNIST.tfrecords'

# how many train/val examples?
num_train_examples = 60_000
num_val_examples = 10_000

x_train = make_dataset(x_train,
                       z_dim,
                       batch_size,
                       shuffle_buffer_size=num_train_examples)

x_val = make_dataset(x_val,
                     z_dim,
                     batch_size,
                     shuffle_buffer_size=num_val_examples)

if TESTING:
    x_train = x_train.take(3)
    x_val = x_val.take(2)

"""
MAKE THE DEMO DATASET
"""

x_demo = tfrecords_filepath + 'x_demo.npy'
z_demo = tfrecords_filepath + 'z_demo.npy'

x_demo = np.load(x_demo)
z_demo = np.load(z_demo)

# z_demo has shape (4,1000,1) and needs to be converted to (4,z_dim,1)
z_demo = z_demo[:,
                0:z_dim,
                :]

# downsample x once to get ground truth, and a second time to get conditioner.
x_demo_down1 = down(x_demo)
x_demo_down2 = down(x_demo_down1)

demo_dataset = None

# Iterate over the 10 classes
for i in range(10):
    # Iterate over the 4 z vectors
    for j in range(4):
        xyz_demo_0 = z_demo[j:j+1]
        xyz_demo_0 = tf.data.Dataset.from_tensor_slices(xyz_demo_0)
        xyz_demo_1 = up(x_demo_down2[i:i+1]).numpy()
        xyz_demo_1 = tf.data.Dataset.from_tensor_slices(xyz_demo_1)
        xyz_demo_2 = x_demo_down1[i:i+1]
        xyz_demo_2 = tf.data.Dataset.from_tensor_slices(xyz_demo_2)

        xyz_demo = tf.data.Dataset.zip((xyz_demo_0,
                                        xyz_demo_1,
                                        xyz_demo_2))

        if not demo_dataset:
            demo_dataset = xyz_demo

        else:
            demo_dataset = demo_dataset.concatenate(xyz_demo)

demo_dataset = demo_dataset.batch(4)

"""
MAKE THE MODELS
"""

generator = make_generator_model(z_dim,
                                 x_shape,
                                 LAYER_NORM,
                                 TESTING,
                                 pretrained_G_filepath)

encoder = make_encoder_model(z_dim,
                             x_shape,
                             DETERMINISTIC,
                             LAYER_NORM,
                             TESTING,
                             pretrained_E_filepath)

critic = make_critic_model(x_shape,
                           critic_input,
                           LAYER_NORM,
                           TESTING,
                           pretrained_C_filepath)

# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam()
encoder_optimizer = tf.keras.optimizers.Adam()
critic_optimizer = tf.keras.optimizers.Adam()

"""
SET CHECKPOINTS
"""

# Save checkpoints
checkpoint_dir = f'./{critic_input_label}_{consistency_label}_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    encoder_optimizer=encoder_optimizer,
    critic_optimizer=critic_optimizer,
    generator=generator,
    encoder=encoder,
    critic=critic)

if ckpt_path:
    status = checkpoint.restore(ckpt_path)

    print(f'CHECKPOINT RESTORED from path {ckpt_path}')
    print('')

"""
DEFINE TRAINING STEPS
"""

@tf.function
def generator_train_step(x,
                         critic_input,
                         DETERMINISTIC,
                         CONSISTENT,
                         alpha):

    """
    Args:
        x: tuple (z, A, B)
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        CONSISTENT: Boolean. Whether or not to include self-consistency terms in the generator loss function.
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        G_loss: The outputs of generator_loss(...) for the generator.

    Updates the generator's weights.
    """

    (z, up_B2, B1) = x

    with tf.GradientTape() as gen_tape:

        # The generated residual.
        R1_gen = generator([z,
                            up_B2])
        # Add instance noise to the fake image.
        if critic_input == 'R':
            R1_gen = instance_noise_R_Rdomain(R1_gen,
                                              alpha)
        else:
            R1_gen = instance_noise_R_Bdomain(R1_gen,
                                              up_B2,
                                              alpha)
            # The full B-domain generated image is then:
            B1_gen = R1_gen + up_B2

        # Add instance noise to the real image.
        if critic_input == 'R':
            B1 = instance_noise_B_Rdomain(B1,
                                          up_B2,
                                          alpha)
        else:
            B1 = instance_noise_B_Bdomain(B1,
                                          alpha)

        # The ground truth residual.
        R1_real = B1 - up_B2

        # The encoded ground truth residual.
        if DETERMINISTIC:
            z_enc = encoder([R1_real,
                             up_B2])
        else:
            z_enc = reparameterize(encoder([R1_real,
                                            up_B2]))

        # The cycled ground truth residual.
        R1_cyc = generator([z_enc,
                            up_B2])

        # Add instance noise to the cycled residual. Noise does enter into this twice, since both B1 and now R1_cyc are noise-ified...but this will improve as the noise is phased out.
        if critic_input == 'R':
            R1_cyc = instance_noise_R_Rdomain(R1_cyc,
                                              alpha)
        else:
            R1_cyc = instance_noise_R_Bdomain(R1_cyc,
                                              up_B2,
                                              alpha)

        # The full B-domain cycled image is then:
        B1_cyc = R1_cyc + up_B2

        # The critic verdict on the generated residual and
        # the critic verdict on the cycled residual.
        if critic_input == 'B':
            verdict_R1_gen = critic(B1_gen)
            verdict_R1_cyc = critic(B1_cyc)

        elif critic_input == 'R':
            verdict_R1_gen = critic(R1_gen)
            verdict_R1_cyc = critic(R1_cyc)

        elif critic_input == 'AB':
            verdict_R1_gen = critic([B1_gen,
                                     up_B2])
            verdict_R1_cyc = critic([B1_cyc,
                                     up_B2])

        # Get the loss for generator.
        G_loss = generator_loss(R1_real,
                                R1_gen,
                                R1_cyc,
                                verdict_R1_gen,
                                verdict_R1_cyc,
                                CONSISTENT,
                                G_loss_weights)

    # Get the total loss.
    total_G_loss = G_loss[0]

    # Now, calculate the gradients for generator.
    G_gradients = gen_tape.gradient(total_G_loss,
                                    generator.trainable_variables)

    # Apply the gradients to generator.
    generator_optimizer.apply_gradients(zip(G_gradients,
                                            generator.trainable_variables))

    return G_loss

@tf.function
def encoder_train_step(x,
                       critic_input,
                       DETERMINISTIC,
                       alpha):

    """
    Args:
        x: tuple (z, A, B)
            NOTE: z, A, and B are all BATCHES of size batch_size. They are NOT individual examples. For example, z has TensorShape([200, 100]) for batch_size=200 and z_dim=100.
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        E_loss: The outputs of encoder_loss(...) for the encoder.

    Updates the encoder's weights.
    """

    (z, up_B2, B1) = x

    with tf.GradientTape() as enc_tape:

        # The generated residual.
        R1_gen = generator([z,
                            up_B2])
        # Add instance noise to the fake image.
        if critic_input == 'R':
            R1_gen = instance_noise_R_Rdomain(R1_gen,
                                              alpha)
        else:
            R1_gen = instance_noise_R_Bdomain(R1_gen,
                                              up_B2,
                                              alpha)

        # The cycled z vector.
        if DETERMINISTIC:
            z_cyc = encoder([R1_gen,
                             up_B2])
        else:
            z_cyc = encoder([R1_gen,
                             up_B2])[0]

        # Add instance noise to the real image.
        if critic_input == 'R':
            B1 = instance_noise_B_Rdomain(B1,
                                          up_B2,
                                          alpha)
        else:
            B1 = instance_noise_B_Bdomain(B1,
                                          alpha)

        # The ground truth residual.
        R1_real = B1 - up_B2

        # The encoded ground truth residual.
        if DETERMINISTIC:
            z_enc = encoder([R1_real,
                             up_B2])
        else:
            z_enc = reparameterize(encoder([R1_real,
                                            up_B2]))

        # Get the loss for encoder.
        E_loss = encoder_loss(z,
                              z_cyc,
                              z_enc,
                              E_loss_weights)

    # Get the total loss.
    total_E_loss = E_loss[0]

    # Now, calculate the gradients for encoder.
    E_gradients = enc_tape.gradient(total_E_loss,
                                    encoder.trainable_variables)

    # Apply the gradients to encoder.
    encoder_optimizer.apply_gradients(zip(E_gradients,
                                          encoder.trainable_variables))

    return E_loss

@tf.function
def critic_train_step(x,
                      critic_input,
                      DETERMINISTIC,
                      alpha):

    """
    Args:
        x: tuple (z, up(B2), B1)
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        C_loss: The outputs of critic_loss(...) for the critic.

    Updates the critic's weights.
    """

    (z, up_B2, B1) = x

    with tf.GradientTape() as crit_tape:

        # The generated residual.
        R1_gen = generator([z,
                            up_B2])
        # Add instance noise to the fake image.
        if critic_input == 'R':
            R1_gen = instance_noise_R_Rdomain(R1_gen,
                                              alpha)
        else:
            R1_gen = instance_noise_R_Bdomain(R1_gen,
                                              up_B2,
                                              alpha)
            # The full B-domain generated image is then:
            B1_gen = R1_gen + up_B2

        # Add instance noise to the real image.
        if critic_input == 'R':
            B1 = instance_noise_B_Rdomain(B1,
                                          up_B2,
                                          alpha)
        else:
            B1 = instance_noise_B_Bdomain(B1,
                                          alpha)

        # The ground truth residual.
        R1_real = B1 - up_B2

        # The encoded ground truth residual.
        if DETERMINISTIC:
            z_enc = encoder([R1_real,
                             up_B2])
        else:
            z_enc = reparameterize(encoder([R1_real,
                                            up_B2]))

        # The cycled ground truth residual.
        R1_cyc = generator([z_enc,
                            up_B2])

        # Add instance noise to the cycled residual. Noise does enter into this twice, since both B1 and now R1_cyc are noise-ified...but this will improve as the noise is phased out.
        if critic_input == 'R':
            R1_cyc = instance_noise_R_Rdomain(R1_cyc,
                                              alpha)
        else:
            R1_cyc = instance_noise_R_Bdomain(R1_cyc,
                                              up_B2,
                                              alpha)

        # The full B-domain cycled image is then:
        B1_cyc = R1_cyc + up_B2

        # Randomly weighted averages for the Wasserstein GP loss term.
        random_average_gen = RandomWeightedAverage(tf.identity(R1_real),
                                                   tf.identity(R1_gen))
        random_average_cyc = RandomWeightedAverage(tf.identity(R1_real),
                                                   tf.identity(R1_cyc))

        # Need these in the B domain, depending on the critic input.
        if critic_input=='B' \
        or critic_input=='AB':
            random_B1_gen = random_average_gen + up_B2
            random_B1_cyc = random_average_cyc + up_B2

        # The critic verdict on the ground truth residual,
        # the critic verdict on the generated residual,
        # the critic verdict on the cycled residual,
        # the critic verdict on the random average of real and gen, and
        # the critic verdict on the random average of real and cyc.
        if critic_input == 'B':
            verdict_R1_real = critic(B1)
            verdict_R1_gen = critic(B1_gen)
            verdict_R1_cyc = critic(B1_cyc)
            verdict_avg_gen = critic(random_B1_gen)
            verdict_avg_cyc = critic(random_B1_cyc)

        elif critic_input == 'R':
            verdict_R1_real = critic(R1_real)
            verdict_R1_gen = critic(R1_gen)
            verdict_R1_cyc = critic(R1_cyc)
            verdict_avg_gen = critic(random_average_gen)
            verdict_avg_cyc = critic(random_average_cyc)

        elif critic_input == 'AB':
            verdict_R1_real = critic([B1,
                                      up_B2])
            verdict_R1_gen = critic([B1_gen,
                                     up_B2])
            verdict_R1_cyc = critic([B1_cyc,
                                     up_B2])
            verdict_avg_gen = critic([random_B1_gen,
                                      up_B2])
            verdict_avg_cyc = critic([random_B1_cyc,
                                      up_B2])

        # Get the loss for critic_2.
        C_loss = critic_loss(verdict_R1_real,
                             verdict_R1_gen,
                             verdict_R1_cyc,
                             random_average_gen,
                             verdict_avg_gen,
                             random_average_cyc,
                             verdict_avg_cyc,
                             C_loss_weights)

        total_C_loss = C_loss[0]

    # Now, calculate the gradients for critic.
    C_gradients = crit_tape.gradient(total_C_loss,
                                     critic.trainable_variables)

    # Apply the gradients to critic.
    critic_optimizer.apply_gradients(zip(C_gradients,
                                         critic.trainable_variables))

    return C_loss

"""
DEFINE EVALUATORS
"""

@tf.function
def generator_eval(x,
                   critic_input,
                   DETERMINISTIC,
                   CONSISTENT,
                   alpha):

    """
    Args:
        x: tuple (z, A, B)
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        CONSISTENT: Boolean. Whether or not to include self-consistency terms in the generator loss function.
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        G_loss: The outputs of generator_loss(...) for the generator.
    """

    (z, up_B2, B1) = x

    # The generated residual.
    R1_gen = generator([z,
                        up_B2])
    # Add instance noise to the fake image.
    if critic_input == 'R':
        R1_gen = instance_noise_R_Rdomain(R1_gen,
                                          alpha)
    else:
        R1_gen = instance_noise_R_Bdomain(R1_gen,
                                          up_B2,
                                          alpha)
        # The full B-domain generated image is then:
        B1_gen = R1_gen + up_B2

    # Add instance noise to the real image.
    if critic_input == 'R':
        B1 = instance_noise_B_Rdomain(B1,
                                      up_B2,
                                      alpha)
    else:
        B1 = instance_noise_B_Bdomain(B1,
                                      alpha)

    # The ground truth residual.
    R1_real = B1 - up_B2

    # The encoded ground truth residual.
    if DETERMINISTIC:
        z_enc = encoder([R1_real,
                         up_B2])
    else:
        z_enc = reparameterize(encoder([R1_real,
                                        up_B2]))

    # The cycled ground truth residual.
    R1_cyc = generator([z_enc,
                        up_B2])

    # Add instance noise to the cycled residual. Noise does enter into this twice, since both B1 and now R1_cyc are noise-ified...but this will improve as the noise is phased out.
    if critic_input == 'R':
        R1_cyc = instance_noise_R_Rdomain(R1_cyc,
                                          alpha)
    else:
        R1_cyc = instance_noise_R_Bdomain(R1_cyc,
                                          up_B2,
                                          alpha)

    # The full B-domain cycled image is then:
    B1_cyc = R1_cyc + up_B2

    # The critic verdict on the generated residual and
    # the critic verdict on the cycled residual.
    if critic_input == 'B':
        verdict_R1_gen = critic(B1_gen)
        verdict_R1_cyc = critic(B1_cyc)

    elif critic_input == 'R':
        verdict_R1_gen = critic(R1_gen)
        verdict_R1_cyc = critic(R1_cyc)

    elif critic_input == 'AB':
        verdict_R1_gen = critic([B1_gen,
                                 up_B2])
        verdict_R1_cyc = critic([B1_cyc,
                                 up_B2])

    # Get the loss for generator.
    G_loss = generator_loss(R1_real,
                            R1_gen,
                            R1_cyc,
                            verdict_R1_gen,
                            verdict_R1_cyc,
                            CONSISTENT,
                            G_loss_weights)

    return G_loss

@tf.function
def encoder_eval(x,
                 critic_input,
                 DETERMINISTIC,
                 alpha):

    """
    Args:
        x: tuple (z, A, B)
            NOTE: z, A, and B are all BATCHES of size batch_size. They are NOT individual examples. For example, z has TensorShape([200, 100]) for batch_size=200 and z_dim=100.
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        E_loss: The outputs of encoder_loss(...) for the encoder.
    """

    (z, up_B2, B1) = x

    # The generated residual.
    R1_gen = generator([z,
                        up_B2])
    # Add instance noise to the fake image.
    if critic_input == 'R':
        R1_gen = instance_noise_R_Rdomain(R1_gen,
                                          alpha)
    else:
        R1_gen = instance_noise_R_Bdomain(R1_gen,
                                          up_B2,
                                          alpha)

    # The cycled z vector.
    if DETERMINISTIC:
        z_cyc = encoder([R1_gen,
                         up_B2])
    else:
        z_cyc = encoder([R1_gen,
                         up_B2])[0]

    # Add instance noise to the real image.
    if critic_input == 'R':
        B1 = instance_noise_B_Rdomain(B1,
                                      up_B2,
                                      alpha)
    else:
        B1 = instance_noise_B_Bdomain(B1,
                                      alpha)

    # The ground truth residual.
    R1_real = B1 - up_B2

    # The encoded ground truth residual.
    if DETERMINISTIC:
        z_enc = encoder([R1_real,
                         up_B2])
    else:
        z_enc = reparameterize(encoder([R1_real,
                                        up_B2]))

    # Get the loss for encoder.
    E_loss = encoder_loss(z,
                          z_cyc,
                          z_enc,
                          E_loss_weights)

    return E_loss

@tf.function
def critic_eval(x,
                critic_input,
                DETERMINISTIC,
                alpha):

    """
    Args:
        x: tuple (z, up(B2), B1)
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        alpha: (1-alpha) is the fraction of instance noise.

    Returns:
        C_loss: The outputs of critic_loss(...) for the critic.
    """

    (z, up_B2, B1) = x

    # The generated residual.
    R1_gen = generator([z,
                        up_B2])
    # Add instance noise to the fake image.
    if critic_input == 'R':
        R1_gen = instance_noise_R_Rdomain(R1_gen,
                                          alpha)
    else:
        R1_gen = instance_noise_R_Bdomain(R1_gen,
                                          up_B2,
                                          alpha)
        # The full B-domain generated image is then:
        B1_gen = R1_gen + up_B2

    # Add instance noise to the real image.
    if critic_input == 'R':
        B1 = instance_noise_B_Rdomain(B1,
                                      up_B2,
                                      alpha)
    else:
        B1 = instance_noise_B_Bdomain(B1,
                                      alpha)

    # The ground truth residual.
    R1_real = B1 - up_B2

    # The encoded ground truth residual.
    if DETERMINISTIC:
        z_enc = encoder([R1_real,
                         up_B2])
    else:
        z_enc = reparameterize(encoder([R1_real,
                                        up_B2]))

    # The cycled ground truth residual.
    R1_cyc = generator([z_enc,
                        up_B2])

    # Add instance noise to the cycled residual. Noise does enter into this twice, since both B1 and now R1_cyc are noise-ified...but this will improve as the noise is phased out.
    if critic_input == 'R':
        R1_cyc = instance_noise_R_Rdomain(R1_cyc,
                                          alpha)
    else:
        R1_cyc = instance_noise_R_Bdomain(R1_cyc,
                                          up_B2,
                                          alpha)

    # The full B-domain cycled image is then:
    B1_cyc = R1_cyc + up_B2

    # Randomly weighted averages for the Wasserstein GP loss term.
    random_average_gen = RandomWeightedAverage(tf.identity(R1_real),
                                               tf.identity(R1_gen))
    random_average_cyc = RandomWeightedAverage(tf.identity(R1_real),
                                               tf.identity(R1_cyc))

    # Need these in the B domain, depending on the critic input.
    if critic_input=='B' \
    or critic_input=='AB':
        random_B1_gen = random_average_gen + up_B2
        random_B1_cyc = random_average_cyc + up_B2

    # The critic verdict on the ground truth residual,
    # the critic verdict on the generated residual,
    # the critic verdict on the cycled residual,
    # the critic verdict on the random average of real and gen, and
    # the critic verdict on the random average of real and cyc.
    if critic_input == 'B':
        verdict_R1_real = critic(B1)
        verdict_R1_gen = critic(B1_gen)
        verdict_R1_cyc = critic(B1_cyc)
        verdict_avg_gen = critic(random_B1_gen)
        verdict_avg_cyc = critic(random_B1_cyc)

    elif critic_input == 'R':
        verdict_R1_real = critic(R1_real)
        verdict_R1_gen = critic(R1_gen)
        verdict_R1_cyc = critic(R1_cyc)
        verdict_avg_gen = critic(random_average_gen)
        verdict_avg_cyc = critic(random_average_cyc)

    elif critic_input == 'AB':
        verdict_R1_real = critic([B1,
                                  up_B2])
        verdict_R1_gen = critic([B1_gen,
                                 up_B2])
        verdict_R1_cyc = critic([B1_cyc,
                                 up_B2])
        verdict_avg_gen = critic([random_B1_gen,
                                  up_B2])
        verdict_avg_cyc = critic([random_B1_cyc,
                                  up_B2])

    # Get the loss for critic_2.
    C_loss = critic_loss(verdict_R1_real,
                         verdict_R1_gen,
                         verdict_R1_cyc,
                         random_average_gen,
                         verdict_avg_gen,
                         random_average_cyc,
                         verdict_avg_cyc,
                         C_loss_weights)

    return C_loss

"""
FUNCTION FOR SAVING IMAGES DURING TRAINING
"""

def generate_and_save_images(epoch,
                             critic_input,
                             alpha,
                             DETERMINISTIC):

    """
    Args:
        epoch: The epoch.
        critic_input: One of 'AB', 'B', or 'R'. The input(s) to the critic.
        alpha: (1-alpha) is the fraction of instance noise.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).

    Saves numpy arrays corresponding to certain tests on examples from the validation dataset. The examples are the first appearing members of each of the 10 classes in the validation set. The generated arrays are:
        Samples of the reconstruction distribution. These are performed using four pre-determined z vectors, for a total of 4 samples per class * 10 classes = 40.
        The encoded ground truth images. There are 10 of these, one per class.
        Cycled ground truth images. There are 10 of these, one per class.
        Cycled z vectors. There are 4 of these per class.
    """

    output_B_gen = []
    output_z_enc = []
    output_B_cyc = []
    output_z_cyc = []

    # Each of the ten batches corresponds to one class.
    for batch in demo_dataset:

        z = tf.cast(batch[0],
                    tf.float32)
        up_B2 = tf.cast(batch[1],
                        tf.float32)
        B1 = tf.cast(batch[2],
                     tf.float32)

        # Only need 1 R_real per class (only z varies in the batch).
        R1_real = B1[0:1] - up_B2[0:1]
        # Add instance noise to the real image.
        if critic_input == 'R':
            R1_real = instance_noise_R_Rdomain(R1_real,
                                               alpha)
        else:
            R1_real = instance_noise_R_Bdomain(R1_real,
                                               up_B2[0:1],
                                               alpha)

        # Generate samples of the reconstruction distribution.
        R1_gen = generator([z,
                            up_B2])

        # Add instance noise to the fake image.
        if critic_input == 'R':
            R1_gen = instance_noise_R_Rdomain(R1_gen,
                                              alpha)
        else:
            R1_gen = instance_noise_R_Bdomain(R1_gen,
                                              up_B2[0:1],
                                              alpha)

        # Move generated residual to B domain.
        B1_gen = R1_gen + up_B2
        output_B_gen.append(B1_gen)

        # Get z_cyc for the 4 original z vectors + this class. It should resemble z.
        # Remember, z is z_dim-dimensional, and in principle the dimensions all have their own distributions; in fact, we are HOPING that the distribution of values in EVERY z_enc is NOT N(0,1), but rather that the collection of ALL N(z_mu, z_var) approximates N(0,1). Rather than exhaustively checking every possibility, we are doing spot checks.
        if DETERMINISTIC:
            z_cyc = encoder([R1_gen,
                             up_B2])
            z_enc = encoder([R1_real,
                             up_B2[0:1]])
        else:
            z_cyc = encoder([R1_gen,
                             up_B2])[0]
            z_enc = reparameterize(encoder([R1_real,
                                            up_B2[0:1]]))

        output_z_cyc.append(z_cyc)
        output_z_enc.append(z_enc)

        # Get R1_cyc for this class.
        R1_cyc = generator([z_enc,
                            up_B2[0:1]])

        # Add instance noise to R1_cyc.
        if critic_input == 'R':
            R1_cyc = instance_noise_R_Rdomain(R1_cyc,
                                              alpha)
        else:
            R1_cyc = instance_noise_R_Bdomain(R1_cyc,
                                              up_B2[0:1],
                                              alpha)

        # Get B1_cyc = R1_cyc + A for this class.
        B1_cyc = R1_cyc + up_B2[0:1]

        output_B_cyc.append(B1_cyc)

    output_B_gen = np.array(output_B_gen)
    output_z_enc = np.array(output_z_enc)
    output_B_cyc = np.array(output_B_cyc)
    output_z_cyc = np.array(output_z_cyc)

    if not TESTING:

        np.save(f'demo_B_gen_{critic_input_label}_{consistency_label}_Epoch{epoch}.npy',
                output_B_gen)
        np.save(f'demo_z_enc_{critic_input_label}_{consistency_label}_Epoch{epoch}.npy',
                output_z_enc)
        np.save(f'demo_B_cyc_{critic_input_label}_{consistency_label}_Epoch{epoch}.npy',
                output_B_cyc)
        np.save(f'demo_z_cyc_{critic_input_label}_{consistency_label}_Epoch{epoch}.npy',
                output_z_cyc)

"""
DEFINE THE TRAINING LOOP
"""

def train_loop(x_train_clean,
               x_val_clean,
               DETERMINISTIC,
               begin_epoch,
               fin_epoch,
               noisy_epochs,
               training_patience,
               critic_patience):

    """
    Args:
        x_train_clean: Training dataset with
            x = (z, A, B)
        x_val_clean: Validation dataset with the same format as x_train.
        DETERMINISTIC: Boolean. Whether or not E is deterministic, outputting a z vector, or variational, outputting a tuple (mu, logvar).
        begin_epoch: Starting epoch. (Can be nonzero if we are continuing training from an earlier run, which is important if fewer than noisy_epochs epochs have been run so far, in addition to being necessary for keeping track of training to date.)
        fin_epoch: Ending epoch. The total number of epochs it is possible to train on here is fin_epoch - begin_epoch. (You may just end up setting fin_epoch to some large number to train until you run out of wall time.)
        noisy_epochs: Number of epochs with some instance noise.  The fraction of the B-domain image that is noisy is 1 at epoch 0 and 0 at epoch noisy_epochs. If noisy_epochs is 0, there is no instance noise.
        training_patience: Stop training if none of the models have improved their results (on x_val) for this many epochs.
        critic_patience: Assume that the critic being trained has converged if its validation loss (on batches of x_val) does not improve for this many batches. (In my experience, just using the already-computed training loss does not result in worse results, but this is the more-rigorous way to do it.)

    ###########################################################################

    HOW INSTANCE NOISE WORKS:

    if epoch < noisy_epochs:
        alpha = epoch / noisy_epochs
    else:
        alpha = 1

    if alpha < 1:
        if critic_input in ['B', 'AB']:
            B -> alpha * B + (1 - alpha) * noise
            R -> alpha * R + (1 - alpha) * (noise - A)
        elif critic_input == 'R':
            B -> alpha * B + (1 - alpha) * (noise + A)
            R -> a * R + (1 - alpha) * noise
        noise = random.uniform(-1,1) (shape is the same as B)
    else:
        B -> B
        R -> R

    Note that because noise is NOT the same between, say, a ground-truth image B and the corresponding reconstruction(s) G(z,A) + A, alpha also represents the fraction of the reconstruction that CAN NOT match the ground truth.
    Note that the exact way in which noise is applied depends on whether the critic wants R-domain or B-domain images as its input!
    The goal of instance noise is to ensure that the support of the two distributions B and G(z,A) is the same. This helps the critic provide useful gradients to the other models. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    By forcing the critic inputs to be completely noisy at first, the supports are forced to be the same; by gradually phasing it out ("annealing") over a large number of epochs, the supports should stay close even as the other models become increasingly good at fooling the critic with their non-noisy output components.
    Note that the conditioner does not need to be noised, for two reasons. First, the average noise over a 2x2 patch of pixels is zero. Thus, down(B + noise) ~ down(B), and each 'pixel' in a noise-ified A tends to have the same value that it would have had without noise. Second, and more importantly, all conditioned models receive the same conditioner A regardless of where their other input came from, so we don't need to worry about different conditioner distributions.

    ###########################################################################

    GENERATOR:

    ||R_cyc - R||_1 | -(1*C(R_gen)) | -(1*C(R_cyc))
    --Want 1st term to be small: R -> E ~> G -> R' gives R'~R.  The L1 distance between random fasion-MNIST images is roughly 0.55 +/- 0.17; the L1 distance between random fasion-MNIST *residuals* is 0.24 +/- 0.05, so anything not substantially smaller than this is bad.  Note that ||B - B'||_1 = ||R - R'||_1; the difference between the two above average distances between random examples is because they are measuring between DIFFERENT examples (specifically, each has a different A, so they don't cancel).  In any case, the ballpark distance should be smaller than ~0.2 (NOTE: varies with the amount of downsampling).
    --Want 2nd term to be negative: G can fool C, given z~N(0,1), resulting in a positive verdict.
    --Want 3rd term to be negative: G can fool C, given z_enc, resulting in a positive verdict.

    if CONSISTENT:
        has 2 additional terms, | consist. (cLR) | consist. (cAE)
        consist. = ||corruption_method(B') - A||_some_norm
        corruption_method(X) = down(X)
        down(B') = down(R' + up(A))
                 = down(R') + down(up(A))
                 = down(R') + A
     -> consist. = ||down(R') + A - A||_2
                 = ||down(R')||_2
        NOTE:  Use L2 norm for consistency loss because we're more interested in heavily penalizing deviations from 0 than we are keeping down(R') sparse, for example.
        NOTE:  This assumes both
            down(X + Y) = down(X) + down(Y) and
            down(up(X)) = X.
            These assumptions both hold for the 2x2-average-pooling corruption method used here, but are NOT true for general corruption methods, in which case the original consistency term must be applied. (General corruption methods have the additional wrinkle of probably not being deterministic.)
        --Want 4th and 5th terms to be small: corrupting the reconstruction reproduces the conditioner, which is the corruption of the ground truth.  For a deterministic corruption, it is in principle possible for these terms to go to 0.

    ###########################################################################

    ENCODER:

    ||z_cyc - z||_1 | KL divergence (cLR) | KL divergence (cAE)
    --Want 1st term to be small: z -> G -> E -> z_cyc gives z_cyc~z.
    --Want 2nd term to be small: E(G(z, up(A)), up(A)) produces vector distributions that are indistinguishable from the normal distribution.
    --Want 3rd term to be small: E(R, up(A)) produces vector distributions that are indistinguishable from the normal distribution.

    ###########################################################################

    CRITIC:

    -(1*C(R)) | -(-1*C(R_gen))/2 | -(-1*C(R_cyc))/2 | GP (cLR) | GP (cAE)
    --Want 1st term to be negative: C can tell when something is real, resulting in a positive verdict.
    --Want 2nd and 3rd terms to be negative: C can tell when something is fake, resulting in a negative verdict.
    --Want 4th and 5th terms to be small: the Lipschitz constraint is satisfied.
        NOTE:  because there are two C(fake) and only one C(real) contributions to the loss, I halve the C(fake) contributions to even out the relative contributions of real and fake verdicts. The 1st term should be close to equal in magnitude (but opposite in sign) to the sum of the 2nd and 3rd terms, which should themselves be virtually identical to one another.
    NOTE:  the consistency loss "should" go here, since (like C) it is a constraint on the outputs of G that requires no comparison with a ground truth - at least in this case, where it can be interpreted as minimizing down(R').  As such, it shows up naturally in the same place as C when training E and G.  However, practically speaking, it was more straightforward to implement as a separate loss in G and E, rather than combining into 1 loss with C.
    """

    # Conditions to end training.
    # Training will be stopped when the TOTAL losses for G, E, and C have ALL plateaued.
    stop_ctr = 0
    epoch = begin_epoch

    # The initial "best loss" is set to infinity so that no matter how bad the first evaluation is, it will be an improvement.
    best_epochal_G_val_loss = np.infty
    best_epochal_E_val_loss = np.infty
    best_epochal_C_val_loss = np.infty

    # These lists will hold the training and validation losses for each epoch.
    G_training_loss_over_time = []
    E_training_loss_over_time = []
    C_training_loss_over_time = []

    G_validation_loss_over_time = []
    E_validation_loss_over_time = []
    C_validation_loss_over_time = []

    # What is/are the distribution term(s) (in E_loss)?
    if DETERMINISTIC:
        distribution_string = f'{LAMBDA_KL_cLR}*KL[batch(z_cyc)||N(0,1)] | {LAMBDA_KL_cAE}*KL[batch(z_enc)||N(0,1)]'
    else:
        distribution_string = f'{LAMBDA_KL_cLR}*KL[N(mu_cyc,var_cyc)||N(0,1)] | {LAMBDA_KL_cAE}*KL[N(mu_enc,var_enc)||N(0,1)]'

    # Which model to start the training loop on?
    model = 'G'

    # The training order will go G -> E -> C* (repeat until converged).
    # *C will be trained until convergence (every time it comes up) using a soft-ish metric, namely that the loss on a batch of the validation dataset has not improved for critic_patience training steps.
    # Whether or not to continue training after a given epoch will be decided using a harder metric, namely that the loss measured on the entire validation set has not improved for training_patience epochs (with training_patience > critic_patience).
    while epoch < fin_epoch \
    and stop_ctr < training_patience:

        # if noisy_epochs > 0 (already checked by instance_noise_alpha()), include instance noise.
        alpha = instance_noise_alpha(tf.identity(epoch * 1.0),
                                     tf.identity(noisy_epochs * 1.0))

        print('--------------------------------------------------')
        print('')
        print(f'Beginning Epoch {epoch} (alpha = {alpha})...')
        print('')
        print('--------------------------------------------------')

        # These lists will store the training losses calculated over batches in this epoch.
        G_loss_over_epoch = []
        E_loss_over_epoch = []
        C_loss_over_epoch = []

        batch_ctr = -1

        for batch in x_train:

            batch_ctr += 1

            if model == 'G':

                G_loss = generator_train_step(batch,
                                              critic_input,
                                              DETERMINISTIC,
                                              CONSISTENT,
                                              alpha)

                G_loss_over_epoch.append(G_loss)

                if CONSISTENT:
                    G_loss_printer = f'{G_loss[0]:.4f} | {G_loss[1]:.4f} | {G_loss[2]:.4f} | {G_loss[3]:.4f} | {G_loss[4]:.4f} | {G_loss[5]:.4f}'

                else:
                    G_loss_printer = f'{G_loss[0]:.4f} | {G_loss[1]:.4f} | {G_loss[2]:.4f} | {G_loss[3]:.4f}'

                print('')
                print('--------------------------------------------------')
                print('')

                print(f'G training loss (Epoch {epoch}, batch {batch_ctr})')

                if CONSISTENT:
                    # Want 2nd, 5th, and 6th columns small; 3rd and 4th columns negative.
                    print(f'total | {LAMBDA_R_RECONSTRUCTION}*||R-R_cyc||_1 | {LAMBDA_CRITIC_cLR}*-(1*C(R_gen)) | {LAMBDA_CRITIC_cAE}*-(1*C(R_cyc)) | {LAMBDA_CONSISTENCY_cLR}*||down(R_gen)||_2 | {LAMBDA_CONSISTENCY_cAE}*||down(R_cyc)||_2')
                else:
                    # Want 2nd column small; 3rd and 4th columns negative.
                    print(f'total | {LAMBDA_R_RECONSTRUCTION}*||R-R_cyc||_1 | {LAMBDA_CRITIC_cLR}*-(1*C(R_gen)) | {LAMBDA_CRITIC_cAE}*-(1*C(R_cyc))')

                print(G_loss_printer)

                model = 'E'

            elif model == 'E':

                E_loss = encoder_train_step(batch,
                                            critic_input,
                                            DETERMINISTIC,
                                            alpha)

                E_loss_over_epoch.append(E_loss)

                E_loss_printer = f'{E_loss[0]:.4f} | {E_loss[1]:.4f} | {E_loss[2]:.4f} | {E_loss[3]:.4f}'

                print('')
                print('--------------------------------------------------')
                print('')

                print(f'E training loss (Epoch {epoch}, batch {batch_ctr})')

                # Want 2nd, 3rd, and 4th columns small.
                if DETERMINISTIC:
                    print(f'total | {LAMBDA_Z_RECONSTRUCTION}*||z-z_cyc||_1 | {distribution_string}')
                else:
                    print(f'total | {LAMBDA_Z_RECONSTRUCTION}*||z-mu_cyc||_1 | {distribution_string}')

                print(E_loss_printer)

                model = 'C'
                first_C_batch = True

            elif model == 'C':

                if first_C_batch:

                    # Initial val loss. It is updated every time critic_eval gives a lower total validation loss.
                    best_C_val_loss = np.infty

                    # Every time best_C_val_loss decreases, the counter resets. If the counter reaches critic_patience, the critic is assumed to have converged.
                    c_ctr = 0

                # Train critic until converged. These steps will be repeated until c_ctr==critic_patience.
                C_loss = critic_train_step(batch,
                                           critic_input,
                                           DETERMINISTIC,
                                           alpha)

                C_loss_over_epoch.append(C_loss)

                C_loss_printer = f'{C_loss[0]:.4f} | {C_loss[1]:.4f} | {C_loss[2]:.4f} | {C_loss[3]:.4f} | {C_loss[4]:.6f} | {C_loss[5]:.6f}'

                if first_C_batch:

                    print('')
                    print('--------------------------------------------------')
                    print('')

                    print(f'C training loss (Epoch {epoch}, batches {batch_ctr}+)')

                    # Want 2nd, 3rd & 4th columns negative, 5th & 6th columns small.
                    print(f'total | -(1*C(R_real)) | -(-1*C(R_gen)/2) | -(-1*C(R_cyc)/2) | {LAMBDA_GP:.0f}*GP (cLR) | {LAMBDA_GP:.0f}*GP (cAE)')
                print(f'Batch {batch_ctr}')
                print(C_loss_printer)

                # Validation loss is estsimated over 10% of the validation data, not the entire validation dataset. Hopefully, this is still large enough to give stable validation results, while not taking as much time as an evaluation over the full validation dataset.
                # Specifically because KL divergence is evaluated over a BATCH, all batches (training and validation) should be the same size to have a fair comparison. As a result, to sample from a larger effective validation batch size, it's easiest to just take multiple batches from the validation set and average their results together.
                C_val_loss = []

                for val_batch in x_val.take(num_val_batches):
                    C_val_loss.append(critic_eval(val_batch,
                                                  critic_input,
                                                  DETERMINISTIC,
                                                  alpha))

                for i in range(num_val_batches):
                    C_val_loss[i] = list(C_val_loss[i])

                C_val_loss = np.array(C_val_loss).mean(axis=0)

                # If the best evaluation changes, reset the counter.
                if C_val_loss[0] < best_C_val_loss:
                    print(f'Best C-convergence validation loss improved from {best_C_val_loss:.4f} to {C_val_loss[0]:.4f}.')
                    best_C_val_loss = C_val_loss[0]
                    c_ctr = 0
                # Otherwise, increment it.
                else:
                    c_ctr += 1

                # End critic training when c_ctr==critic_patience.
                if c_ctr == critic_patience:
                    model = 'G'

                else:
                    model = 'C'
                    first_C_batch = False

        # An epoch has been finished.
        print('')
        print('--------------------------------------------------')
        print('')

        print(f'Epoch {epoch} complete.')
        print('Evaluating epochal validation losses...')

        print('')

        # Get the training losses over that epoch.
        G_training_loss_over_time.append(
            np.array(G_loss_over_epoch).mean(axis=0))
        E_training_loss_over_time.append(
            np.array(E_loss_over_epoch).mean(axis=0))
        C_training_loss_over_time.append(
            np.array(C_loss_over_epoch).mean(axis=0))

        # And the validation losses.
        G_val_loss = []
        E_val_loss = []
        C_val_loss = []

        # The output here is a list of tuples of tensors, one tuple per batch...
        for val_batch in x_val:

            G_val_loss.append(generator_eval(val_batch,
                                             critic_input,
                                             DETERMINISTIC,
                                             CONSISTENT,
                                             alpha))
            E_val_loss.append(encoder_eval(val_batch,
                                           critic_input,
                                           DETERMINISTIC,
                                           alpha))
            C_val_loss.append(critic_eval(val_batch,
                                          critic_input,
                                          DETERMINISTIC,
                                          alpha))

        # ...so convert these to arrays and take the mean over the batch axis.
        for i in range(len(G_val_loss)):
            G_val_loss[i] = list(G_val_loss[i])
            E_val_loss[i] = list(E_val_loss[i])
            C_val_loss[i] = list(C_val_loss[i])

        G_val_loss = np.array(G_val_loss).mean(axis=0)
        E_val_loss = np.array(E_val_loss).mean(axis=0)
        C_val_loss = np.array(C_val_loss).mean(axis=0)

        G_validation_loss_over_time.append(G_val_loss)
        E_validation_loss_over_time.append(E_val_loss)
        C_validation_loss_over_time.append(C_val_loss)

        # If the best evaluation for ANY of G, E, or C changes, reset the counter for this level's training.
        reset = False

        if epoch < noisy_epochs:
            print('Instance noise still annealing; Epochal validation loss is being recorded, but not used to determine early stopping.')
            reset = True

        if epoch >= noisy_epochs \
        and G_val_loss[0] < best_epochal_G_val_loss:
            print(f'Best Epochal G validation loss updated from {best_epochal_G_val_loss:.4f} to {G_val_loss[0]:.4f}.')
            best_epochal_G_val_loss = G_val_loss[0]
            reset = True
        elif epoch >= noisy_epochs \
        and G_val_loss[0] >= best_epochal_G_val_loss:
            print(f'Best Epochal G validation loss has not changed ({G_val_loss[0]:.4f} > {best_epochal_G_val_loss:.4f})...')

        if epoch >= noisy_epochs \
        and E_val_loss[0] < best_epochal_E_val_loss:
            print(f'Best Epochal E validation loss updated from {best_epochal_E_val_loss:.4f} to {E_val_loss[0]:.4f}.')
            best_epochal_E_val_loss = E_val_loss[0]
            reset = True
        elif epoch >= noisy_epochs \
        and E_val_loss[0] >= best_epochal_E_val_loss:
            print(f'Best Epochal E validation loss has not changed ({E_val_loss[0]:.4f} > {best_epochal_E_val_loss:.4f})...')

        if epoch >= noisy_epochs \
        and C_val_loss[0] < best_epochal_C_val_loss:
            print(f'Best Epochal C validation loss updated from {best_epochal_C_val_loss:.4f} to {C_val_loss[0]:.4f}.')
            best_epochal_C_val_loss = C_val_loss[0]
            reset = True
        elif epoch >= noisy_epochs \
        and C_val_loss[0] >= best_epochal_C_val_loss:
            print(f'Best Epochal C validation loss has not changed ({C_val_loss[0]:.4f} > {best_epochal_C_val_loss:.4f})...')

        if reset:
            stop_ctr = 0
            if epoch < noisy_epochs:
                print('Instance noise still annealing; stop_ctr has not been incremented.')
            else:
                print('Validation loss has not plateaued for all models at this level.  Resetting stop_ctr to 0...')
            print('')
        # Otherwise, increment it.
        else:
            stop_ctr += 1
            print(f'stop_ctr = {stop_ctr} out of {training_patience}.')
            print('')

        # Save numpy arrays containing:
        # 1) Samples from the the generated B distribution;
        # 2) The ground truth images, encoded into latent space;
        # 3) The cycled ground truth images;
        # 4) The cycled latent space vectors.
        generate_and_save_images(epoch,
                                 critic_input,
                                 alpha,
                                 DETERMINISTIC)

        if  SAVE_INTERVAL > 0 \
        and not (epoch + 1) % SAVE_INTERVAL:

            # Save the trained models to this point.
            generator.save(f'G_{critic_input_label}_{consistency_label}.hdf5')
            encoder.save(f'E_{critic_input_label}_{consistency_label}.hdf5')
            critic.save(f'C_{critic_input_label}_{consistency_label}.hdf5')

            # Save the training losses to this point.
            np.save(f'G_train_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(G_training_loss_over_time))
            np.save(f'E_train_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(E_training_loss_over_time))
            np.save(f'C_train_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(C_training_loss_over_time))

            # Save the validation losses to this point.
            np.save(f'G_val_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(G_validation_loss_over_time))
            np.save(f'E_val_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(E_validation_loss_over_time))
            np.save(f'C_val_loss_{critic_input_label}_{consistency_label}.npy',
                    np.asarray(C_validation_loss_over_time))

        # Save a checkpoint every 10 epochs.
        if not (epoch + 1) % 10:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # Increment the epoch counter
        epoch += 1

###############################################################################

start_time = time.time()

train_loop(x_train,
           x_val,
           DETERMINISTIC,
           begin_epoch,
           fin_epoch,
           noisy_epochs,
           training_patience,
           critic_patience)

end_time = time.time()
time_diff = end_time - start_time
print('')
print(f'Total time to train ({fin_epoch - begin_epoch} epochs) is {time_diff}s.')