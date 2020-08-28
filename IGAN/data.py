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

def sample_Data_by_Count(train_images,train_labels,sample_breakdown):
    new_train_images=[]
    new_train_labels=[]
    true_breakdown=np.zeros(10)

    #Find those samples
    for j in range(len(train_labels)):
        if sample_breakdown[train_labels[j]]!=0:
            sample_breakdown[train_labels[j]]-=1
            true_breakdown[train_labels[j]]+=1
            new_train_images.append(train_images[j])
            new_train_labels.append(train_labels[j])
        
        if np.sum(sample_breakdown==0):
            break
        
    tb=true_breakdown
        
    return tb ,np.array(new_train_images),np.array(new_train_labels)

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

def sample_Data_IVOM(test_images,test_labels,samples):
    new_train_images=[[] for i in range(10) ]
    
    target_samples=np.ones(10)*samples

    if (np.sum(target_samples!=0)):
        #Find those samples
        for j in range(len(test_labels)):
            if target_samples[test_labels[j]]!=0:
                target_samples[test_labels[j]]-=1
                new_train_images[test_labels[j]].append(test_images[j]) 


        
    return new_train_images

def sample_Data_by_Count_IVOM(train_images,train_labels,sample_breakdown):
    new_train_images=[[] for i in range(10) ]


    if (np.sum(sample_breakdown!=0)):
        #Find those samples
        for j in range(len(train_labels)):
            if sample_breakdown[train_labels[j]]!=0:
                sample_breakdown[train_labels[j]]-=1
                new_train_images[train_labels[j]].append(train_images[j]) 
        
    return new_train_images

# train_images,train_labels= load_data_from_tf()
# total_samples=60000
# sample_breakdown=[0.3,0.2,0.1,0.1,0.1,0.08,0.06,0.058,0.001,0.001]
# true_breakdown,new_train_images,new_train_labels=sample_Data(train_images,train_labels,total_samples,sample_breakdown)
            
#Test sample_data_IVOM
# test_images,test_labels= load_test_data_from_tf()
# test=sample_Data_IVOM(test_images,test_labels,2)           
    

#Test Sample Breakdown by count
train_images,train_labels= load_data_from_tf()
# sample_breakdown=[1000,1000,500,500,300,300,200,200,10,10]
sample_breakdown=[4000,4000,4000,10,4000,4000,20,30,4000,4000]
true_breakdown,new_train_images,new_train_labels=sample_Data_by_Count(train_images,train_labels,sample_breakdown)
          