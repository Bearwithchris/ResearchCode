# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:56:18 2020

@author: Chris
"""

from PIL import Image
import numpy as np
import os,sys
from sklearn.utils import shuffle 
import tensorflow as tf
import random

dir1 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/Male'
dir2 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/female'


#Test Data Prep
size=[227,227]

def loadImgs_Test(dir_images):
    cap=20
    count=0
    imagelist=[]
    images_paths=os.listdir(dir_images)
    count=0
    for i in range(len(images_paths)):   
        if count==cap:
            break
        else:
            images=Image.open(dir_images+"/"+random.choice(images_paths))
            images=np.asarray(images)
            images=(images- 127.5) / 127.5
            images=tf.image.resize(images,size)
            imagelist.append(images)
            count+=1
    return np.array(imagelist)
def prep_dataset_Test(directory_male,directory_female):
    images_male=loadImgs_Test(directory_male)
    images_male_labels=np.ones(len(images_male))
    images_female=loadImgs_Test(directory_female)
    images_female_labels=np.zeros(len(images_female))
    images_concat=np.concatenate((images_male,images_female),axis=0)
    images_labels_concat=np.concatenate((images_male_labels,images_female_labels))
    images_concat_shuffled,images_labels_concat_shuffled=shuffle(images_concat,images_labels_concat)
    return images_concat_shuffled,images_labels_concat_shuffled


#Training Data Prep
def loadImgs(dir_images,cap=5000):
    # cap=5000
    count=0
    imagelist=[]
    images_paths=os.listdir(dir_images)
    for i in images_paths:
        if count==cap:
            break
        else:
            images=Image.open(dir_images+"/"+i)
            images=np.asarray(images)
            images=(images- 127.5) / 127.5
            images=tf.image.resize(images,size)
            imagelist.append(images)
            count+=1
    return np.array(imagelist)

def loadImgs_random(dir_images,cap=1000):
    # cap=5000
    imagelist=[]
    images_paths=os.listdir(dir_images)
    for i in range(int(cap)):
        images=Image.open(dir_images+"/"+random.choice(images_paths))
        images=np.asarray(images)
        images=(images- 127.5) / 127.5
        images=tf.image.resize(images,size)
        imagelist.append(images)

    return np.array(imagelist)

def prep_dataset(directory_male,directory_female):
    images_male=loadImgs(directory_male)
    images_male_labels=np.ones(len(images_male))
    images_female=loadImgs(directory_female)
    images_female_labels=np.zeros(len(images_female))
    images_concat=np.concatenate((images_male,images_female),axis=0)
    images_labels_concat=np.concatenate((images_male_labels,images_female_labels))
    images_concat_shuffled,images_labels_concat_shuffled=shuffle(images_concat,images_labels_concat)
    return images_concat_shuffled,images_labels_concat_shuffled

#Bias and ref dataset prep========================================================================================
def prep_bias_data(directory_male,directory_female,samples,bias=0.5):
    female_samples=samples*bias
    male_samples=samples*(1.0-bias)
    
    #Load_bias_data
    images_male=loadImgs_random(directory_male,male_samples)
    images_female=loadImgs_random(directory_female,female_samples)
    images_concat=np.concatenate((images_male,images_female),axis=0)
    images_concat=shuffle(images_concat)
    
    return images_concat

#If evaluation is True must have gamma=1.0
def datasets(dir1,dir2,samples,bias,gamma=1.0,evaluation=True):
    bias_data_samples=samples/(gamma+1)
    ref_data_samples=samples-bias_data_samples
    
    bias=prep_bias_data(dir1,dir2,bias_data_samples,bias)
    bias_labels=np.ones(len(bias))
    
    ref=prep_bias_data(dir1,dir2,ref_data_samples)
    ref_labels=np.zeros(len(ref))
    
    
    if evaluation==False:
        images_labels_concat=np.concatenate((ref_labels,bias_labels))
        images_concat=np.concatenate((ref,bias),axis=0)
        
        images_concat,images_labels_concat=shuffle(images_concat,images_labels_concat)
    else:
        batch=32
        batchsets=len(bias)/batch
        for i in range(int(batchsets)):
            if i==0:        
                images_labels_concat=np.concatenate((ref_labels[0:batch],bias_labels[0:batch]))
                images_concat=np.concatenate((ref[0:batch],bias[0:batch]),axis=0)
            else:
                images_labels_temp=np.concatenate((ref_labels[i*batch:(i+1)*batch],bias_labels[i*batch:(i+1)*batch]))
                images_temp=np.concatenate((ref[i*batch:(i+1)*batch],bias[i*batch:(i+1)*batch]),axis=0)
                images_labels_concat=np.concatenate((images_labels_concat,images_labels_temp))
                images_concat=np.concatenate((images_concat,images_temp),axis=0)
        


    
    return images_labels_concat,images_concat

labels,images=datasets(dir1,dir2,1000,bias=0.9)
# images_concat_shuffled,images_labels_concat_shuffled=prep_dataset(dir1,dir2)