# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:03:37 2020

@author: Chris
"""
import tensorflow as tf 
import numpy as np
import model_tools as mt
import matplotlib.pyplot as plt

# output_activation = tf.keras.activations.linear
output_activation = tf.keras.activations.tanh
kernel_initializer = 'he_normal'
NOISE_DIM = 512
# image_size = 128

def generator_input_block(x):
    '''
        Generator input block
    '''
    x = mt.EqualizeLearningRate(tf.keras.layers.Dense(4*4*512, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_dense')(x)
    x = mt.PixelNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)
    x = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(512, 3, strides=1, padding='same',
                                          kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_conv2d')(x)
    x = mt.PixelNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x

def build_4x4_generator(noise_dim=NOISE_DIM):
    '''
        4 * 4 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    # Not used in 4 * 4, put it here in order to keep the input here same as the other models
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    
    rgb_out = to_rgb(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=rgb_out)
    return model

def build_8x8_generator(noise_dim=NOISE_DIM):
    '''
        8 * 8 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_16x16_generator(noise_dim=NOISE_DIM):
    '''
        16 * 16 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_32x32_generator(noise_dim=NOISE_DIM):
    '''
        32 * 32 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_64x64_generator(noise_dim=NOISE_DIM):
    '''
        64 * 64 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_128x128_generator(noise_dim=NOISE_DIM):
    '''
        128 * 128 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = mt.upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_256x256_generator(noise_dim=NOISE_DIM):
    '''
        256 * 256 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = mt.upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = mt.upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=128, filters=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_512x512_generator(noise_dim=NOISE_DIM):
    '''
        512 * 512 Generator
    '''
    # Initial block
    inputs = tf.keras.layers.Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = tf.keras.layers.Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = mt.upsample_block(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = mt.upsample_block(x, in_filters=512, filters=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = mt.upsample_block(x, in_filters=256, filters=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    x, _ = mt.upsample_block(x, in_filters=128, filters=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    ########################
    # Fade in block
    ########################
    x, up_x = mt.upsample_block(x, in_filters=64, filters=32, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(512, 512))
    
    previous_to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    to_rgb = mt.EqualizeLearningRate(tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(512, 512))
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    ########################
    # Left branch in the paper
    ########################
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=combined)
    return model



def discriminator_block(x):
    '''
        Discriminator output block
    '''
    x = mt.MinibatchSTDDEV()(x)
    x = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, 3, strides=1, padding='same',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_1')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, 4, strides=1, padding='valid',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_2')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x =tf.keras.layers. Flatten()(x)
    x = mt. EqualizeLearningRate(tf.keras.layers. Dense(1, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_dense')(x)
    return x

def build_4x4_discriminator():
    '''
        4 * 4 Discriminator
    '''
    inputs = tf.keras.layers. Input((4,4,3))
    # Not used in 4 * 4
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    # From RGB
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    x = from_rgb(inputs)
    x = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='conv2d_up_channel')(x)
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_8x8_discriminator():
    '''
        8 * 8 Discriminator
    '''
    fade_in_channel = 512
    inputs = tf.keras.layers. Input((8,8,3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable block
    ########################
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_16x16_discriminator():
    '''
        16 * 16 Discriminator
    '''
    fade_in_channel = 512
    inputs = tf.keras.layers. Input((16, 16, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_32x32_discriminator():
    '''
        32 * 32 Discriminator
    '''
    fade_in_channel = 512
    inputs = tf.keras.layers. Input((32, 32, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_64x64_discriminator():
    '''
        64 * 64 Discriminator
    '''
    fade_in_channel = 512
    inputs = tf.keras.layers. Input((64, 64, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=256, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_128x128_discriminator():
    '''
        128 * 128 Discriminator
    '''
    fade_in_channel = 256
    inputs = tf.keras.layers. Input((128, 128, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
   
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=128, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_256x256_discriminator():
    '''
        256 * 256 Discriminator
    '''
    fade_in_channel = 128
    inputs = tf.keras.layers. Input((256, 256, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=64, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = mt.downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_512x512_discriminator():
    '''
        512 * 512 Discriminator
    '''
    fade_in_channel = 64
    inputs = tf.keras.layers. Input((512, 512, 3))
    alpha = tf.keras.layers. Input((1), name='input_alpha')
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = mt. EqualizeLearningRate(tf.keras.layers.Conv2D(32, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(512, 512))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = mt.downsample_block(r_x, filters1=32, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(512,512))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = mt.downsample_block(x, filters1=64, filters2=128, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    x = mt.downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = mt.downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = mt.downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = tf.keras.Model(inputs=[inputs, alpha], outputs=x)
    return model


#Model Builder

def model_builder(target_resolution):
    '''
        Helper function to build models
    '''
    generator = None
    discriminator = None
    if target_resolution == 4:
        generator = build_4x4_generator()
        discriminator = build_4x4_discriminator()
    elif target_resolution == 8:
        generator = build_8x8_generator()
        discriminator = build_8x8_discriminator()
    elif target_resolution == 16:
        generator = build_16x16_generator()
        discriminator = build_16x16_discriminator()
    elif target_resolution == 32:
        generator = build_32x32_generator()
        discriminator = build_32x32_discriminator()
    elif target_resolution == 64:
        generator = build_64x64_generator()
        discriminator = build_64x64_discriminator()
    elif target_resolution == 128:
        generator = build_128x128_generator()
        discriminator = build_128x128_discriminator()
    elif target_resolution == 256:
        generator = build_256x256_generator()
        discriminator = build_256x256_discriminator()
    elif target_resolution == 512:
        generator = build_512x512_generator()
        discriminator = build_512x512_discriminator()
    else:
        print("target resolution models are not defined yet")
    return generator, discriminator

# generator, discriminator = model_builder(image_size)
# generator.summary()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# discriminator.summary()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)