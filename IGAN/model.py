# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:22:01 2020

@author: Chris
"""

import tensorflow as tf
import numpy as np

class model():
    def __init__(self):
        self.gen_input_shape=[100,]
        self.disc_input_shape=[28,28,1]
        
    def make_generator(self):
        inp=tf.keras.layers.Input(shape=self.gen_input_shape)
        x=tf.keras.layers.Dense(7*7*256 ,use_bias=False)(inp)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        
        x=tf.keras.layers.Reshape((7,7,256))(x)
        
        x=tf.keras.layers.Conv2DTranspose(128,(5,5),strides=(1,1), padding="same",use_bias=False)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        
        x=tf.keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2), padding="same",use_bias=False)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        
        x=tf.keras.layers.Conv2DTranspose(1,(5,5),strides=(2,2), padding="same",use_bias=False)(x)

        x=tf.keras.Model(inputs=inp,outputs=x)
        tf.keras.utils.plot_model(x , to_file='Generator.png', show_shapes=True, dpi=64) #Added to visualise model
        return x
    
    def make_discriminator(self):
        inp=tf.keras.layers.Input(shape=self.disc_input_shape)
        # y=tf.keras.layers.Conv2D(64,(5,5), strides=(2,2), padding='same')(inp)
        y=tf.keras.layers.Conv2D(128,(5,5), strides=(2,2), padding='same')(inp)
        y=tf.keras.layers.LeakyReLU()(y)
        y=tf.keras.layers.Dropout(0.3)(y)
        
        # y=tf.keras.layers.Conv2D(128,(5,5), strides=(2,2), padding='same')(y)
        y=tf.keras.layers.Conv2D(256,(5,5), strides=(2,2), padding='same')(y)
        y=tf.keras.layers.LeakyReLU()(y)
        y=tf.keras.layers.Dropout(0.3)(y)
        
        y=tf.keras.layers.Flatten()(y)
        y=tf.keras.layers.LeakyReLU()(y)
        y=tf.keras.layers.Dropout(0.3)(y)
        
        y=tf.keras.layers.Dense(1)(y)
        
        y=tf.keras.Model(inputs=inp,outputs=y)
        tf.keras.utils.plot_model(y , to_file='Discriminator.png', show_shapes=True, dpi=64) #Added to visualise mode
        
        return y
        

        

        
        
    