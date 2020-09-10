# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:11:15 2020

@author: Chris
"""
import tensorflow as tf 
import numpy as np
import model_tools as mt
import matplotlib.pyplot as plt
import model as m
from tensorflow.keras import backend as K
import os
from functools import partial
import data
import math
import time
from IPython.display import clear_output

import config as c

#Configure Gpus
c.config_gpu()
# c.precision() #WIP



batch_size = 32
CURRENT_EPOCH = 1 # Epoch start from 1. If resume training, set this to the previous model saving epoch.

#Directories
DATA_BASE_DIR="H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Train/alt"
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", 'train_data') # Sets up a log directory.
MODEL_PATH = 'models'
OUTPUT_PATH = 'outputs'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

# output_activation = tf.keras.activations.linear
output_activation = tf.keras.activations.tanh
kernel_initializer = 'he_normal'
NOISE_DIM = 512
image_size = 4
LR = 1e-3
BETA_1 = 0.
BETA_2 = 0.99
EPSILON = 1e-8

# Decay learning rate
MIN_LR = 0.000001
DECAY_FACTOR=1.00004



#Initiliase Data
list_ds = tf.data.Dataset.list_files(DATA_BASE_DIR + '/*')                    #Returns a tensor Dataset of file directory
preprocess_function = partial(data.preprocess_image, target_size=image_size)  #Partially fill in a function data.preprocess_image with the arguement image_size
train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds


#Initilaise Model
generator, discriminator = m.model_builder(image_size)
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


#Define Optimiser
D_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)


def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, D_optimizer, G_optimizer):
    '''
        Set new learning rate to optimizersz
    '''
    K.set_value(D_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)
    
def calculate_batch_size(image_size):
    # if image_size < 64:
    #     return 16
    # elif image_size < 128:
    #     return 12
    # elif image_size == 128:
    #     return 8
    # elif image_size == 256:
    #     return 4
    # else:
    #     return 3
    if image_size <= 16:
        return 16
    elif image_size <= 32:
        return 8
    else:
        return 4
    
    
    
def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    # Test input is a list include noise and label
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, '{}x{}_image_at_epoch_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))
    plt.show()

num_examples_to_generate = 9

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
# sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
# sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)
# generate_and_save_images(generator, 0, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=False, is_flatten=False)


LAMBDA = 10


#==================TRAINING Functions======================================================================
#@tf.function
def WGAN_GP_train_d_step(generator, discriminator, real_image, alpha, batch_size, step):
    '''
        One training step
        
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator([noise, alpha], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator([fake_image_mixed, alpha], training=True)
            
        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
        
        fake_pred = discriminator([fake_image, alpha], training=True)
        real_pred = discriminator([real_image, alpha], training=True)
        
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    D_optimizer.apply_gradients(zip(D_gradients,discriminator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)
            
#@tf.function
def WGAN_GP_train_g_step(generator, discriminator, alpha, batch_size, step):
    '''
        One training step
        
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator([noise, alpha], training=True)
        fake_pred = discriminator([fake_image, alpha], training=True)
        G_loss = -tf.reduce_mean(fake_pred)
    # Calculate the gradients for discriminator
    G_gradients = g_tape.gradient(G_loss,generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,generator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('G_loss', G_loss, step=step)



#Load old Models trained
# Load previous resolution model
if image_size > 4:
    if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2)))):
        generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
        print("generator loaded")
    if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2)))):
        discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
        print("discriminator loaded")
        
# # To resume training, comment it if not using.
# if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size)))):
#     generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size))), by_name=False)
#     print("generator loaded")
# if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size)))):
#     discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size))), by_name=False)
#     print("discriminator loaded")


#===================================Actual Training======================================================================
total_data_number = len(os.listdir(DATA_BASE_DIR))
switch_res_every_n_epoch = 40


current_learning_rate = LR
training_steps = math.ceil(total_data_number / batch_size)
# Fade in half of switch_res_every_n_epoch epoch, and stablize another half
alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
alpha = min(1., (CURRENT_EPOCH - 1) % switch_res_every_n_epoch * training_steps *  alpha_increment)
EPOCHs = 320
SAVE_EVERY_N_EPOCH = 5 # Save checkpoint at every n epoch

sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)



for epoch in range(CURRENT_EPOCH, EPOCHs + 1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))
    print('Current alpha: %f' % (alpha,))
    print('Current resolution: {} * {}'.format(image_size, image_size))
    # Using learning rate decay
#     current_learning_rate = learning_rate_decay(current_learning_rate)
#     print('current_learning_rate %f' % (current_learning_rate,))
#     set_learning_rate(current_learning_rate) 
    
    for step, (image) in enumerate(train_data):
        current_batch_size = image.shape[0]
        alpha_tensor = tf.constant(np.repeat(alpha, current_batch_size).reshape(current_batch_size, 1), dtype=tf.float32)
       
        # Train step
        WGAN_GP_train_d_step(generator, discriminator, image, alpha_tensor,
                             batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
        WGAN_GP_train_g_step(generator, discriminator, alpha_tensor,
                             batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
        
        
        # update alpha
        alpha = min(1., alpha + alpha_increment)
        
        #Loading screen
        if step % 10 == 0:
            print ('.', end='')
    
    # # Clear jupyter notebook cell output
    # clear_output(wait=True)
    
    # Using a consistent image (sample_X) so that the progress of the model is clearly visible.
    generate_and_save_images(generator, epoch, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)
    
    #Override old model and save over it
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_size)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_size)))
        print ('Saving model for epoch {}'.format(epoch))
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch,time.time()-start))
    
    
    # Train next resolution
    if epoch % switch_res_every_n_epoch == 0:
        #Save weights one more time
        print('saving {} * {} model'.format(image_size, image_size))
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_size)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_size)))
        
        # Reset alpha
        alpha = 0
        
        #Save old image size
        previous_image_size = int(image_size)
        #Incerement to next image_size
        image_size = int(image_size * 2)
        
        if image_size > 512:
            print('Resolution reach 512x512, finish training')
            break
        
        #Load new model with increment dimension 
        print('creating {} * {} model'.format(image_size, image_size))
        generator, discriminator = m.model_builder(image_size)
        generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
        discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
        
        #Reprocessing new data in new size
        print('Making {} * {} dataset'.format(image_size, image_size))
        batch_size = calculate_batch_size(image_size)
        preprocess_function = partial(data.preprocess_image, target_size=image_size)
        train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)
        training_steps = math.ceil(total_data_number / batch_size)
        alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
        print('start training {} * {} model'.format(image_size, image_size))
        