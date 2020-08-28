# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:18:25 2020

@author: Chris
"""
import model 
import loss as l
import data
from IMLE import IMLE_fn2

import tensorflow as tf
import time 
from IPython import display
import matplotlib.pyplot as plt
import os

EPOCHS = 200 # was 150
noise_dim = 100
num_examples_to_generate = 16


#Load models
m=model.model()
generator=m.make_generator()
discriminator=m.make_discriminator()


#load optimisers
generator_optimizer = tf.keras.optimizers.Adam(1e-4) #was 1e-4
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) # was 1e-4

#Load loss functions
generator_loss=l.loss_fn_searcher("generator_LSGAN")
discriminator_loss=l.loss_fn_searcher("discriminator_LSGAN")
    
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
  
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      #Adversarial loss
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      
      # #Rec loss
      # lamb=36 #was 0.4
      # rec_loss=IMLE_fn2.IMLE_Pixel(generated_images,new_train_images,new_train_labels,20,[3,6,7])
      # gen_loss=gen_loss+rec_loss
            
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    for epoch in range(epochs):
      start = time.time()
    
      for image_batch in dataset:
        train_step(image_batch)
    
      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epoch + 1,
                               seed)
    
      # Save the model every 15 epochs
      if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)
  
if __name__=="__main__":
    #Load MNIST data
    BUFFER_SIZE = 60000
    # sample_breakdown=[0.3,0.2,0.1,0.1,0.1,0.08,0.06,0.058,0.001,0.001]
    
    # sample_breakdown_count=[4000,4000,3000,3000,2000,2000,1,1,0,0]
    sample_breakdown_count=[4000,4000,4000,10,4000,4000,20,30,4000,4000]
    
    train_images,train_labels= data.load_data_from_tf()
    # true_breakdown,new_train_images,new_train_labels=data.sample_Data(train_images,train_labels,BUFFER_SIZE,sample_breakdown)
    true_breakdown,new_train_images,new_train_labels=data.sample_Data_by_Count(train_images,train_labels,sample_breakdown_count)


    BATCH_SIZE = 256 # was 256
    
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(new_train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    #Train
    train(train_dataset, EPOCHS)
    print ("True Breakdown: "+str(true_breakdown))


#Test mode collapse
def mode_collapse_figures(epoch):
    for i in range (10):
        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        epoch+=1
        generate_and_save_images(generator,epoch,seed)

mode_collapse_figures(EPOCHS)
    