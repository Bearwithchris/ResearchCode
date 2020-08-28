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
def loadImgs(dir_images):
    cap=5000
    count=0
    imagelist=[]
    images_paths=os.listdir(dir_images)
    count=0
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

def prep_dataset(directory_male,directory_female):
    images_male=loadImgs(directory_male)
    images_male_labels=np.ones(len(images_male))
    images_female=loadImgs(directory_female)
    images_female_labels=np.zeros(len(images_female))
    images_concat=np.concatenate((images_male,images_female),axis=0)
    images_labels_concat=np.concatenate((images_male_labels,images_female_labels))
    images_concat_shuffled,images_labels_concat_shuffled=shuffle(images_concat,images_labels_concat)
    return images_concat_shuffled,images_labels_concat_shuffled

# images_concat_shuffled,images_labels_concat_shuffled=prep_dataset(dir1,dir2)