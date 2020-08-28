# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:51:27 2020

@author: Chris
"""
import data

import tensorflow as tf
# import tensorflow_probability as tfp
import os
import model
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import copy
import time

# Load data
# train_images,train_labels= data.load_test_data_from_tf()

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

def load_model(checkpoint_path):
    #Optimisers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    #Models
    generator,discriminator = create_model()
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    return generator, discriminator, generator_optimizer,discriminator_optimizer


#New aproach
# 1) Generate from a z-input
# 2) Loss function will be the output of that (z-input) and the x chosen
# 3) Back propagate and update the z-input instead of the weights


def rec_loss(fake_output,target):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # diff=np.sqrt(np.square((target-prediction)))
    diff=tf.math.reduce_mean(tf.math.square(tf.math.subtract(target,fake_output)))

    # diff = cross_entropy(tf.zeros_like(diff), diff)
    # diff=tf.reduce_sum(diff)
    
    return diff


#Train teh IVOM
def train_step_fn():
    @tf.function
    def step(generator,generator_optimizer,z,target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          #Generate image
          generated_images = generator(z, training=False)
    
          #reconstruction loss
          loss = rec_loss(generated_images,target)
          
          
          
        gradients_of_generator = gen_tape.gradient(loss,[z])
        # gradients_of_generator = gen_tape.gradient(loss,z)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, [z]))
        return loss
    return step

# Plot the IVOM image comparison    
def plotReults(generator,target,z_orig,z,cat,sample,epoch,plot_dir):
    plt.subplot(1, 3, 1)
    plt.imshow(target[0,:,:,0])
    plt.subplot(1, 3, 2)
    plt.imshow(generator(z_orig,training=False)[0,:,:,0])
    plt.subplot(1, 3, 3)
    plt.imshow(generator(z,training=False)[0,:,:,0])
    plt.savefig(plot_dir+str(cat)+"_sample"+str(sample)+"_epoch_"+str(epoch))
    # plt.show()

#Train and get the IVOM image    

epoch=20000





def train(generator,generator_optimizer,target,z_orig,z,cat,sample,plot_dir="./plots/"):
    train_step=train_step_fn()
    for i in range(epoch):
        # print (i)
        z_old=copy.deepcopy(z)
        loss=train_step(generator,generator_optimizer,z,target)
        if (i%1000==0):
            plotReults(generator,target,z_orig,z,cat,sample,i,plot_dir)
            print ("Cat: "+str(cat)+" sample:"+str(sample)+" Epoch: "+str(i)+ "Loss:"+str(loss))
    return loss

def plt_dist(cp_vanilla,save_file="./plots/"):
    #Temp mean Calculations
    mean_loss=[]
    for i in range(len(cp_vanilla)):
        sumLoss=0
        
        for j in range(len(cp_vanilla[0])):
            sumLoss+=cp_vanilla[i][j]
        sumLoss=sumLoss/samples
        mean_loss.append(sumLoss)

    mean_loss=np.array(mean_loss)
    x=np.linspace(0,9,10)
    plt.ylabel("IVOM Loss")
    plt.xlabel("MNIST class label")
    plt.bar(x,mean_loss)
    plt.savefig(save_file+"data_dist")
    end_time=time.time()-start_time        

if __name__=="__main__":
    start_time=time.time()
    samples=10   #Define the number of step per class
    
    #Load test data
    test_images,test_labels= data.load_test_data_from_tf()
    sample_breakdown_count=[4000,4000,4000,10,4000,4000,20,30,4000,4000]
    # test_sample_images=data.sample_Data_IVOM(test_images,test_labels,samples)  
    test_sample_images=data.sample_Data_by_Count_IVOM(test_images,test_labels,sample_breakdown_count)
    
    checkpoint_path0 = './training_checkpoints'
    checkpoint_path1 = './training_checkpoints_vanilla'
    checkpoint_path2 = './training_checkpoints_IMLE'
    
    #Define training modes
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
 #Test IVOM score on all classes for a specfied number of samples   
    def train_with_checkpoint(path):
        #Inititialise training step fn
        
        cat_IVOM_score=[]   #Ivom score per category 
        catNum=0
        for cat in test_sample_images:
            sample_IVOM_score=[]
           
            
            targetIter = tf.data.Dataset.from_tensor_slices(cat).batch(BATCH_SIZE)
            targetIter=iter(targetIter)
            for s in range(samples):
                #Initilaise generator 
                generator,__,generator_optimizer,__=load_model(path)
                #Initialise the noise vector 
                z = tf.random.normal([BATCH_SIZE, noise_dim])
                z_orig=copy.deepcopy(z)
                z = tf.Variable(z,trainable=True)
                
                target=next(targetIter)
                loss=train(generator,generator_optimizer,target,z_orig,z,catNum,s)
                sample_IVOM_score.append(loss.numpy())
            cat_IVOM_score.append(sample_IVOM_score)
            catNum+=1
        return cat_IVOM_score

#Test IVOM score with specific taget number on two different path
    def train_with_checkpoint_specific(target_num,path1,path2):
        #Inititialise training step fn
        ivom_score_path1=[]
        ivom_score_path2=[]
        
        #Filter specific cat and create a tensor
        cat=test_sample_images[target_num]
        targetIter = tf.data.Dataset.from_tensor_slices(cat).batch(BATCH_SIZE)
        targetIter=iter(targetIter)
        
        for s in range(samples):
            #Initialise the noise vector fir both paths 
            z = tf.random.normal([BATCH_SIZE, noise_dim])
            z_orig=copy.deepcopy(z)
            z = tf.Variable(z,trainable=True)
            
            
            #Initiliase_target
            target=next(targetIter)
            
            #Path 1 test
            generator,__,generator_optimizer,__=load_model(path1)
            loss=train(generator,generator_optimizer,target,z_orig,z,0,s,plot_dir="./comparison_plots/vanilla_plot/")
            ivom_score_path1.append(loss.numpy())
            
            #Path 2 test
            generator2,__,generator_optimizer2,__=load_model(path2)
            loss2=train(generator2,generator_optimizer2,target,z_orig,z,0,s,plot_dir="./comparison_plots/imle_plot/")
            ivom_score_path2.append(loss2.numpy())
    
            
        return ivom_score_path1,ivom_score_path2
    
    def barplot_versus(vanilla_score,IMLE_score,sample):
        data=[]
        data.append(vanilla_score)
        data.append(IMLE_score)
        #Plot versus samples 
        X = np.arange(0,sample)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(X + 0.00, data[0], color = 'g', width = 0.25)
        ax.bar(X + 0.25, data[1], color = 'r', width = 0.25) 
        ax.set_title("IVOM Score")
        ax.legend(labels=['Vanilla', 'LSGAN'])
        ax.set_xlabel("Sample number")
        ax.set_ylabel("IVOM score")
         
        fig = plt.figure()
        ax = fig.add_axes(
        [0,0,1,1])
        x = ["vanilla" , "LSGAN"]
        y = np.array([np.mean(np.array(vanilla_score)),np.mean(np.array(IMLE_score))])
        ax.bar(x,y)
        plt.savefig("./comparison_plots/Graph_comparisons/")
        plt.show()
        
    # cp_vanilla=train_with_checkpoint(checkpoint_path0)
    # cp_IMLE=train_with_checkpoint(checkpoint_path2)
    # plt_dist(cp_vanilla)
    vanilla_score,IMLE_score=train_with_checkpoint_specific(1,"./training_checkpoints_050820_vanilla","./training_checkpoints")
    barplot_versus(vanilla_score,IMLE_score,samples)
