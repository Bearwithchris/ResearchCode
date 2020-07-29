# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:12:48 2020

@author: Chris
"""

from sklearn.neighbors import KDTree
import numpy as np
import tensorflow as tf

import data

def flattern(data):
    
    temp=[]
    
    #Extract the sample and flattern 
    for i in range(len(data)):
        temp.append(np.reshape(data[i],[28*28]))
    
    temp=np.array(temp)
    
    return (temp)

def flattern_sample(data,ss):
    #sample data
    data_index=[i for i in range(len(data))]
    sample_index=np.random.choice(data_index,ss)
    temp=[]
    
    #Extract the sample and flattern 
    for i in range(len(sample_index)):
        temp.append(np.reshape(data[sample_index[i]],[28*28]))
    
    temp=np.array(temp)
    
    return (temp)

# train_images,train_labels=data.load_data_from_tf()
# sample=flattern_sample(train_images,10)


def IMLE_Pixel(generated_data,data,sample):
    #Take sample from the data and flattern
    sample_data=flattern_sample(data,sample)
    
    #Generated Data
    flattern_generated_data=flattern(generated_data)
    # Lrec=0
    
    nn=[]
    
    #Cycle though each data point and find the nearest generated figure 
    for i in range(len(sample_data)):
        temp=flattern_generated_data
        temp_sample_data=np.reshape(sample_data[i],[1,28*28])
        temp=np.concatenate((temp_sample_data,temp),axis=0)

        
        #Knn-tree
        kdt=KDTree(temp,metric="euclidean")
        rankDist=kdt.query(temp,k=2,return_distance=False)
        nearest_arg=rankDist[0][1]
        
        #L1 distance
        # dist=np.sum(np.abs(temp_sample_data-flattern_generated_data[nearest_arg]))
        # Lrec+=dist
        try:
            nn.append(flattern_generated_data[nearest_arg-1])
        except:
            print ('error')
            break
    
    mae=tf.keras.losses.MeanAbsoluteError()
    mae=mae(sample_data,np.array(nn))
        
    return (mae)
        
        
        
train_images,train_labels= data.load_data_from_tf()
mae=IMLE_Pixel(train_images[0:20],train_images[30:50],10)

# mae=mae.numpy()