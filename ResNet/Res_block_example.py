# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:12:11 2020
https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
tensorboard command: "tensorboard --logdir=log --port 6006"
@author: Chris
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt

#Import CIFAR data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Shuffle randomly select of a batch for 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)) #Normalised the X data
train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y)) # Crop the x Data
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))# Randomly augment the data
train_dataset = train_dataset.repeat()

#Shuffle randomly select of a batch for 5000
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)) #Normalised the X data
valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y)) #Crop the X Data
valid_dataset = valid_dataset.repeat() #Randomly Augment

#ResNet Model
inputs = keras.Input(shape=(24, 24, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)

#Filter size 64, Kernal size=3x3
def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x) #Batch Norm perform after first conv
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x) #Batch Norm Performed after second Conv
  x = layers.Add()([x, input_data])  #Addition of the identity block 
  x = layers.Activation('relu')(x)
  return x

def non_res_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(x)
  x = layers.BatchNormalization()(x)
  return x

#Resnet
num_res_net_blocks = 10
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

#Last Conv + Global Average Pooling + Cifar 10 calssification
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

#Non res model
y_inputs = keras.Input(shape=(24, 24, 3))
y = layers.Conv2D(32, 3, activation='relu')(y_inputs)
y = layers.Conv2D(64, 3, activation='relu')(y)
y = layers.MaxPooling2D(3)(y)

num_non_res_net_blocks = 10
for i in range(num_non_res_net_blocks):
    y = non_res_block(y, 64, 3)

#Last Conv + Global Average Pooling + Cifar 10 calssification
y = layers.Conv2D(64, 3, activation='relu')(y)
y = layers.GlobalAveragePooling2D()(y)
y = layers.Dense(256, activation='relu')(y)
y = layers.Dropout(0.5)(y)
y_outputs = layers.Dense(10, activation='softmax')(y)


res_net_model = keras.Model(inputs, outputs)
non_res_net_model = keras.Model(y_inputs, y_outputs)


res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

non_res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='.\\log\{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
]
res_net_model.fit(train_dataset, epochs=30, steps_per_epoch=195,
          validation_data=valid_dataset,
          validation_steps=3, callbacks=callbacks)

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='.\\log\{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
]

non_res_net_model.fit(train_dataset, epochs=30, steps_per_epoch=195,
          validation_data=valid_dataset,
          validation_steps=3, callbacks=callbacks)