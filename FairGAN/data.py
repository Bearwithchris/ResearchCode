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
from functools import partial

dir1 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/male'
dir2 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/female'


#Test Data Prep
size=[224,224]

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
    # imagelist=np.array([])
    # imagelist=[]
    images_paths=os.listdir(dir_images)
    for i in range(int(cap)):
        images=Image.open(dir_images+"/"+random.choice(images_paths))
        images=np.asarray(images)
        images=(images- 127.5) / 127.5
        images=tf.image.resize(images,size)
        # imagelist.append(images)
        if i==0:
            imagelist=np.reshape(images,[1,images.shape[0],images.shape[1],images.shape[2]])
        else:
            imagelist=np.vstack((imagelist,np.reshape(images,[1,images.shape[0],images.shape[1],images.shape[2]])))

    return imagelist

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
        batch=int(batch/2)
        batchsets=len(bias)/batch
        for i in range(int(batchsets)):
            if i==0:        
                images_labels_concat=np.concatenate((ref_labels[0:batch],bias_labels[0:batch]))
                images_concat=np.concatenate((ref[0:batch],bias[0:batch]),axis=0)
                images_labels_concat,images_concat=shuffle(images_labels_concat,images_concat)
            else:
                images_labels_temp=np.concatenate((ref_labels[i*batch:(i+1)*batch],bias_labels[i*batch:(i+1)*batch]))
                images_temp=np.concatenate((ref[i*batch:(i+1)*batch],bias[i*batch:(i+1)*batch]),axis=0)
                images_labels_temp,images_temp=shuffle(images_labels_temp,images_temp)
                
                images_labels_concat=np.concatenate((images_labels_concat,images_labels_temp))
                images_concat=np.concatenate((images_concat,images_temp),axis=0)
        


    
    return images_labels_concat,images_concat

# labels,images=datasets(dir1,dir2,1000,bias=0.9)
# images_concat_shuffled,images_labels_concat_shuffled=prep_dataset(dir1,dir2)


#=========Prep Data Reduce Load==============================================================================================
def normalize(image):
    '''
        normalizing the images to [-1, 1]
    '''
    image = tf.cast(image, tf.float32) 
    # image = tf.cast(image, tf.float16) #Trying mix precision
    image = (image - 127.5) / 127.5
    return image

def augmentation(image):
    '''
        Perform some augmentation
    '''
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image(file_path, target_size=512):
    images = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.resize(images, (target_size, target_size),
                            method='nearest', antialias=True)
    images = augmentation(images)
    images = normalize(images)
    return images

def loadImgs_random_V2(dir_images,target_size=32,cap=1000):
    list_ds=tf.data.Dataset.list_files(dir_images + '/*') 
    preprocess_function = partial(preprocess_image, target_size=32)  #Partially fill in a function data.preprocess_image with the arguement image_size
    list_ds=list_ds.take(cap)
    # train_data = list_ds.map(preprocess_function).shuffle(100)  #Apply the function pre_process to list_ds
    train_data = list_ds.map(preprocess_function) #Apply the function pre_process to list_ds

    return train_data #Return back a data_set

def prep_bias_data_V2(directory_male,directory_female,samples,bias=0.5):
    female_samples=samples*bias
    male_samples=samples*(1.0-bias)
    
    #Load_bias_data
    images_male=loadImgs_random_V2(directory_male,male_samples)
    images_female=loadImgs_random_V2(directory_female,female_samples)
    
    images_concat=images_male.concatenate(images_female)
    # images_concat=images_concat.shuffle(100)
    
    return images_concat

def datasets_V2(dir1,dir2,samples,bias,gamma=1.0,evaluation=True):
    bias_data_samples=samples/(gamma+1)
    ref_data_samples=samples-bias_data_samples
    
    bias=prep_bias_data_V2(dir1,dir2,bias_data_samples,bias)
    bias_labels=tf.data.Dataset.from_tensors(np.ones(int(bias_data_samples)))
    
    ref=prep_bias_data_V2(dir1,dir2,ref_data_samples)
    ref_labels=tf.data.Dataset.from_tensors(np.zeros(int(ref_data_samples)))
    
    
    if evaluation==False:
        images_labels_concat=ref_labels.concatenate(bias_labels)
        images_concat=ref.concatenate(bias)
        
        # images_concat,images_labels_concat=shuffle(images_concat,images_labels_concat)
    else:
        batch=10
        
        bias=bias.batch(batch)
        bias_labels=bias_labels.batch(batch)
        
        ref=ref.batch(batch)
        ref_labels=ref_labels.batch(batch)
        
        images_concat=tf.data.Dataset.zip((ref, bias)).map(lambda x, y: tf.concat((x, y), axis=0))
        images_labels_concat=tf.data.Dataset.zip((ref_labels, bias_labels)).map(lambda x, y: tf.concat((x, y), axis=0))

    
    return images_labels_concat,images_concat

# labels,images=datasets_V2(dir1,dir2,100,bias=0.9)