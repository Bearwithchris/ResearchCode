# -*- coding: utf-8 -*-
"""
https://www.tensorflow.org/tutorials/generative/dcgan
Created on Sun Jun  7 13:40:07 2020

@author: Chris
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, AveragePooling2D, Dense, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

#PreProcessing Data
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images=train_images.astype('float32')
# train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 128
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 4
seed = tf.random.normal([num_examples_to_generate, noise_dim])

  
def build_discriminator(image_shape):

    model = Sequential()
    # normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))


    model.summary()
    return model



def build_generator(z_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    
    model.summary()
    
    return model

#Initialising Generator 
z_dim=100
generator = build_generator(z_dim)

#Test Generator Function
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, :])

#Initilaise Disciminaotr
image_shape=[32,32,3]
discriminator = build_discriminator(image_shape)

#Test Discriminator
decision = discriminator(generated_image)
print (decision)

    
#################################################
#Loss
################################################
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#Define optimisers
generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)




###################################################################################
#Utility code
###################################################################################

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False) #Test Model

  #Plot model 
  fig = plt.figure(figsize=(4,4)) 

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((np.asarray(predictions[i, :, :, :]*255).astype(np.uint8)))
      # plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


################################################
#Training
###############################################


#`tf.function` - Compiles a function in a callable tensorflow Graph 
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    #tf.gradientTape - Record operation for automatic differntiation 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True) #Generate Image

      real_output = discriminator(images, training=True) #Discrimate real output 
      fake_output = discriminator(generated_images, training=True) #Discriminate fake output

      gen_loss = generator_loss(fake_output) #Ability to fakeout the discriminator log(1-D(G(z)))
      disc_loss = discriminator_loss(real_output, fake_output) #Ability to identify fake and truth 

    #Differentiate w.r.t to trainable variables
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) 
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    #Apply optimiser "apply_gradient class" on the zip-ed gradient trainable variable pairs"
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return (gen_loss,disc_loss)
    
def train(dataset, epochs):
    
#Train per epoch 
  for epoch in range(epochs):
    start = time.time()
    
    count=0
    #Mini-batch training 
    for image_batch in dataset:
      gLoss,dLoss=train_step(image_batch)
      print ("Epoch: %f , %f/%f Generator Loss: %f, Discriminator Loss %f"%(epoch,count,BATCH_SIZE,gLoss,dLoss))
      count+=1

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,epoch + 1,seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                            epochs,
                            seed)

# Train model
train(train_dataset, EPOCHS)

# # Generate Gif
# # Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(EPOCHS)