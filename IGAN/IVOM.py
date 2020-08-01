# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:51:27 2020

@author: Chris
"""
import data

import tensorflow as tf
import tensorflow_probability as tfp
import os
import model
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import copy

# Load data
train_images,train_labels= data.load_test_data_from_tf()

BUFFER_SIZE = 60000
BATCH_SIZE = 1 # was 256
noise_dim = 100
num_examples_to_generate=1

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
def create_model():
    m=model.model()
    return (m.make_generator(),m.make_discriminator())

def load_model():
    #Optimisers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    #Models
    generator,discriminator = create_model()
    
    checkpoint_path = './training_checkpoints_vanilla'
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    return generator, discriminator, generator_optimizer,discriminator_optimizer

#======================================================================================
# Archieved 
# ** This approach is problematic as our z output and loss are different in dimensions
#=======================================================================================
# optimizer = tf.keras.optimizers.Adam(1e-4)

# def mse(init_z):  
#     #Generated
#     prediction= model(init_z, training=False)
    
#     # diff=np.sqrt(np.square((target-prediction)))
#     diff=tf.keras.losses.mse(target,prediction)
#     return diff

# # @tf.function
# def SGD(model,init_z,target):

#     optim_results = tfp.optimizer.lbfgs_minimize(mse,initial_position=init_z,num_correction_pairs=10,tolerance=1e-8)
           
#     return optim_results
# if __name__=="__main__":     
#     model,__=load_model()

#     #Generated
#     init_z=tf.random.normal([num_examples_to_generate, noise_dim])

#     #Target data
#     target = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#     target=next(iter(target))
    
    
#     optim_results=SGD(model,init_z,target)
    
#================================================================================================


#New aproach
# 1) Generate from a z-input
# 2) Loss function will be the output of that (z-input) and the x chosen
# 3) Back propagate and update the z-input instead of the weights


def rec_loss(fake_output,target):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # diff=np.sqrt(np.square((target-prediction)))
    diff=tf.keras.losses.mse(target,fake_output)
    # diff = cross_entropy(tf.zeros_like(diff), diff)
    diff=tf.reduce_sum(diff)
    
    return diff


#Train teh IVOM
@tf.function
def train_step(z,target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      #Generate image
      generated_images = generator(z, training=False)

      #reconstruction loss
      loss = rec_loss(generated_images,target)
      
      
      
    gradients_of_generator = gen_tape.gradient(loss,[z])
    # gradients_of_generator = gen_tape.gradient(loss,z)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, [z]))
    return (loss)

#Plot the IVOM image comparison    
def plotReults(target,z_orig,z):
    plt.subplot(1, 3, 1)
    plt.imshow(target[0,:,:,0])
    plt.subplot(1, 3, 2)
    plt.imshow(generator(z_orig,training=False)[0,:,:,0])
    plt.subplot(1, 3, 3)
    plt.imshow(generator(z,training=False)[0,:,:,0])
    plt.show()

#Train and get the IVOM image    
epoch=20000
def train(target,z_orig,z):
    for i in range(epoch):
        z_old=copy.deepcopy(z)
        loss=train_step(z,target)
        if (i%100==0):
            plotReults(target,z_orig,z)
            print (loss)
        

if __name__=="__main__":
    optimizer = tf.keras.optimizers.Adam(1e-4)    
    z = tf.random.normal([BATCH_SIZE, noise_dim])
    z_orig=copy.deepcopy(z)
    z = tf.Variable(z,trainable=True)
    generator,__,generator_optimizer,__=load_model()
    
    target = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    target=next(iter(target))
    train(target,z_orig,z)
