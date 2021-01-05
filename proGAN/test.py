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



batch_size = 16
CURRENT_EPOCH = 159 # Epoch start from 1. If resume training, set this to the previous model saving epoch.

#Directories
# DATA_BASE_DIR="../../scratch/alt"
#DATA_BASE_DIR="D:/GIT/ResearchCode/proGAN/alt"
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
image_size = 32
LR = 1e-3
BETA_1 = 0.
BETA_2 = 0.99
EPSILON = 1e-8

# Decay learning rate
MIN_LR = 0.000001
DECAY_FACTOR=1.00004



#Initiliase Data
# list_ds = tf.data.Dataset.list_files(DATA_BASE_DIR + '/*')                    #Returns a tensor Dataset of file directory
# preprocess_function = partial(data.preprocess_image, target_size=image_size)  #Partially fill in a function data.preprocess_image with the arguement image_size
# train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)  #Apply the function pre_process to list_ds


#Initilaise Model
generator, discriminator = m.model_builder(image_size)
generator.summary()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator.summary()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


# #Define Optimiser
# D_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
# G_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)


    
    
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



#Load old Models trained
# Load previous resolution model
# if image_size > 4:
#     if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2)))):
#         generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
#         print("generator loaded")
#     if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2)))):
#         discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_size / 2))), by_name=True)
#         print("discriminator loaded")
        
# To resume training, comment it if not using.
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size)))):
    generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("generator loaded")
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size)))):
    discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_size))), by_name=False)
    print("discriminator loaded")


#===================================Actual Training======================================================================
# total_data_number = len(os.listdir(DATA_BASE_DIR))
# switch_res_every_n_epoch = 40


# current_learning_rate = LR
# training_steps = math.ceil(total_data_number / batch_size)
# # Fade in half of switch_res_every_n_epoch epoch, and stablize another half
# alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
# alpha = min(1., (CURRENT_EPOCH - 1) % switch_res_every_n_epoch * training_steps *  alpha_increment)
EPOCHs = 320
SAVE_EVERY_N_EPOCH = 5 # Save checkpoint at every n epoch

sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)




# Using a consistent image (sample_X) so that the progress of the model is clearly visible.
generate_and_save_images(generator, CURRENT_EPOCH, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)
    
