# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:00:35 2020

@author: Chris
"""

import numpy as np
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



    
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
        
def loss_fn_searcher(lf):
    if (lf=="discriminator"):
        return discriminator_loss
    elif (lf=="generator"):
        return generator_loss
    else:
        print ("Invalid loss function")
        

generator_loss=loss_fn_searcher("generator")