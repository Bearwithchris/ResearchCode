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

#Sample a specific amount of data from the set
def sample_fn(data,ss):
    #sample data
    data_index=[i for i in range(len(data))]
    sample_index=np.random.choice(data_index,ss)
    temp=[]
    
    #Extract the sample and flattern 
    for i in range(len(sample_index)):
        temp.append(data[sample_index[i]])
    
    temp=np.array(temp)
    
    return (temp)


# train_images,train_labels=data.load_data_from_tf()
# sample=flattern_sample(train_images,10)

#Filter only the minority data as specified
def filter_cat(data,data_labels,cat):
    new_data=[]
    data_labels=np.array(data_labels)
    for i in (cat):
        arg=np.where(data_labels==i)[0]
        for j in arg:
            new_data.append(data[j])
    return np.array(new_data)
        
        
        
        

def IMLE_Pixel(generated_data,data,data_labels,sample,cat):
    # Take sample from the data and flattern
    data=filter_cat(data,data_labels,cat)
    sample_data=sample_fn(data,sample)
    # distance_total=tf.Variable(0,dtype="float32")
    distance_total=0
    distance_total=tf.convert_to_tensor(distance_total,dtype="float32")
    for sample_data_one in sample_data:
        sample_data_one=np.reshape(sample_data_one,[1,28,28,1])
        
        #Make a replicate sample to the same dimensions as the generate
        tf_sample_data_one=tf.convert_to_tensor(sample_data_one, dtype="float32")
        replicated_sample=tf.repeat(tf_sample_data_one, repeats=[generated_data.shape[0]],axis=0)
        
        #Find the distance between sample and the generated data
        distance=tf.abs(tf.add(replicated_sample,tf.negative(generated_data)))
        
        distance=tf.keras.backend.min(tf.reduce_mean(tf.reduce_mean(distance,axis=1),1))

        distance_total=tf.add(tf.math.multiply(tf.constant(1/sample_data.shape[0]),distance),distance_total)
    

        
        
    return distance_total
        # sample_data=tf.Variable(ample_data)
    
    
    # nn=[]
    
    # #Cycle though each data point and find the nearest generated figure 
    # for i in range(len(sample_data)):
    #     emp=flattern_generated_datat
    #     temp_sample_data=np.reshape(sample_data[i],[1,28*28])
    #     temp=np.concatenate((temp_sample_data,temp),axis=0)

        
    #     #Knn-tree
    #     kdt=KDTree(temp,metric="euclidean")
    #     rankDist=kdt.query(temp,k=2,return_distance=False)
    #     nearest_arg=rankDist[0][1]
        
    #     #L1 distance
    #     # dist=np.sum(np.abs(temp_sample_data-flattern_generated_data[nearest_arg]))
    #     # Lrec+=dist
    #     try:
    #         nn.append(flattern_generated_data[nearest_arg-1])
    #     except:
    #         print ('error')
    #         break
    
    # mae=tf.keras.losses.MeanAbsoluteError()
    # mae=mae(sample_data,np.array(nn))
        
    return 0
        
        
        
# train_images,train_labels= data.load_data_from_tf()
# mae=IMLE_Pixel(tf.Variable(train_images[0:20]),train_images[30:50],10)

# mae=mae.numpy()