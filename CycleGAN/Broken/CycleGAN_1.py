# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:51:07 2020

@author: Chris
"""


import tensorflow as tf
import tensorflow_datasets as tfds

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset, metadata = tfds.load('cycle_gan/horse2zebra',with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

#============================================================================================
#                                 Image Handler Start
#============================================================================================

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

#############Apply transformation to dataset with map function 
# num_parallel_calls is for perfromance sake
# cache is for memory optimisation reasons 
train_horses = train_horses.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_zebras = train_zebras.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)


sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)

plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)



#============================================================================================
#                                 Image Handler End
#============================================================================================






#========================================================================================
#                               Neural Networks Functions Start
#=======================================================================================

#Downsample block weight initilisation + Convo layer + batch norm + L-RELU
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

#Upsample Inv_Transpose + BatchNorm + Dropout +Relu
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

#Generator Model===========================================================================================
def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)                                             #Normal Downsample
    skips.append(x)                                         #Append the skip output for later use in the UNET

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)                                               #Normal Upsample 
    x = tf.keras.layers.Concatenate()([x, skip])            #Implementing U-NET foraward pass of the skip network via concat of the skip X

  x = last(x)                                               #output iamge

  return tf.keras.Model(inputs=inputs, outputs=x)

# LAMBDA = 100

# def generator_loss(disc_generated_output, gen_output, target):
#   #Generator GAN Loss
#   gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

#   # mean absolute error
#   l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

#   total_gen_loss = gan_loss + (LAMBDA * l1_loss)

#   return total_gen_loss, gan_loss, l1_loss

#Discriminator model============================================================================================

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  #Take in two iamges 1) The target image (drawing form), 2) the generated image 
  x = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  
  # inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  # tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
  # x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=x, outputs=last)

# #PatchGAN takes the 30x30 output and binary cross entropy it 
# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
# def discriminator_loss(disc_real_output, disc_generated_output):
    
#   #PatchGAN Real
#   real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)                     #Real output and one tuple

#   #PatchGAN fake
#   generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

#   total_disc_loss = real_loss + generated_loss

#   return total_disc_loss

#image generator 

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


#Gen test demo
OUTPUT_CHANNELS = 3

generator_g = Generator()
generator_f = Generator()
discriminator_x= Discriminator()
discriminator_y= Discriminator()



#=============================================================================
#               Pix2pix Image demo Start
#============================================================================

# to_zebra = generator_g(sample_horse)
# to_horse = generator_f(sample_zebra)
# plt.figure(figsize=(8, 8))
# contrast = 8

# imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
# title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

# for i in range(len(imgs)):
#   plt.subplot(2, 2, i+1)
#   plt.title(title[i])
#   if i % 2 == 0:
#     plt.imshow(imgs[i][0] * 0.5 + 0.5)
#   else:
#     plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

# plt.figure(figsize=(8, 8))

# plt.subplot(121)
# plt.title('Is a real zebra?')
# plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

# plt.subplot(122)
# plt.title('Is a real horse?')
# plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

# plt.show()

#=============================================================================
#               Pix2pix Image demo End
#============================================================================

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            generator_f=generator_f,
                            discriminator_x=discriminator_x,
                            discriminator_y=discriminator_y,
                            generator_g_optimizer=generator_g_optimizer,
                            generator_f_optimizer=generator_f_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
  
  
  
######################################################################################
#                                Training
#####################################################################################

EPOCHS = 40

def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    #Given X- Horse and Y-Zebra
    
    #==================== Generator =====================================#
    
    # Generator G translates X -> Y (Fake Translated Image, Zebra)
    fake_y = generator_g(real_x, training=True)
    # Generator F translates Y -> X (Reversed Fake Translated Image, Horse)
    cycled_x = generator_f(fake_y, training=True)

    # Generator G translates X -> Y (Fake Translated Image, Horse)
    fake_x = generator_f(real_y, training=True)
    # Generator F translates Y -> X (Reversed Fake Translated Image, Zebra)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    #==================== Discriminator =================================#
    #Determine which is the real X (Horse)
    disc_real_x = discriminator_x(real_x, training=True)
    disc_fake_x = discriminator_x(fake_x, training=True)
    
    #Determine which the real Y (Zebra)
    disc_real_y = discriminator_y(real_y, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    #=================== Calculating Loss ===============================#
    # calculate the loss of the generator ( via BCE )
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    # Calculate Cycle Loss (Via L1 Norm loss )
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss + Identity loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
    
    #Discriminator loss
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)
  
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                              discriminator_y.trainable_variables)
  
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))
  
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
  
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))
if __name__=="__main__":
    for epoch in range(EPOCHS):
      start = time.time()
    
      n = 0
      for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
          print ('.', end='')
        n+=1
    
      clear_output(wait=True)
      # Using a consistent image (sample_horse) so that the progress of the model
      # is clearly visible.
      generate_images(generator_g, sample_horse)
    
      if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                              ckpt_save_path))
    
      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))
      