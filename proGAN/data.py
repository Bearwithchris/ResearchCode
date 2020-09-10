# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:16:55 2020

@author: Chris
"""
import tensorflow as tf

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