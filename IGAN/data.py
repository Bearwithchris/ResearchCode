# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:39:21 2020

@author: Chris
"""

import tensorflow as tf
import numpy as np

def load_data_from_tf():
    #Load MNIST data
    (train_images, train_labels),(_,_)=tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    
    return (train_images,train_labels)

def load_test_data_from_tf():
    #Load MNIST data
    (train_images, train_labels),(test_images,test_labels)=tf.keras.datasets.mnist.load_data()
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    
    return (test_images,test_labels)


def sample_Data(train_images,train_labels,total_samples,sample_breakdown):
    new_train_images=[]
    new_train_labels=[]
    true_breakdown=np.ones(10)
    #Check if input arguments are acceptable:\
    if (total_samples<=0 or total_samples>60000 or len(sample_breakdown)!=10):
        print ("!!!!!!!!!!!!!    Invalid Arguement  !!!!!!!!")
        return (0,0,0)
    #Check that the sample breakdown adds to 100%
    if np.sum(sample_breakdown)!=1.0:
        print ("!!!!!!!!!!!!!    Invalid sample breakdown  !!!!!!!!")
        return (0,0,0)
    
    else:
        #sample count
        for i in range(len(sample_breakdown)):
            sample_breakdown[i]=sample_breakdown[i]*total_samples
            
        #Find those samples
        for j in range(len(train_labels)):
            if sample_breakdown[train_labels[j]]!=0:
                sample_breakdown[train_labels[j]]-=1
                true_breakdown[train_labels[j]]+=1
                new_train_images.append(train_images[j])
                new_train_labels.append(train_labels[j])
            
            if np.sum(total_samples==0):
                break
            
        tb=true_breakdown/np.sum(true_breakdown)
        
    return tb ,np.array(new_train_images),np.array(new_train_labels)

train_images,train_labels= load_data_from_tf()
total_samples=60000
sample_breakdown=[0.18,0.13,0.14,0.12,0.12,0.11,0.10,0.09,0.005,0.005]
true_breakdown,new_train_images,new_train_labels=sample_Data(train_images,train_labels,total_samples,sample_breakdown)
            
        
            
    
    