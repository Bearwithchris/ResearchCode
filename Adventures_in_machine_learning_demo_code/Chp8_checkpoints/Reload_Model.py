# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:16:03 2020

@author: Chris
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime as dt
import os 

#Prepare training data=======================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

#Data Augmentation
train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))

train_dataset = train_dataset.repeat()


#Prepare testing data======================================================
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
valid_dataset = valid_dataset.repeat()

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(96, 3, padding='same', activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(l=0.001),input_shape=(24, 24, 3)))
    model.add(keras.layers.Conv2D(96, 3, 2, padding='same', activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(l=0.001)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(192, 3, padding='same', activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(l=0.001)))
    model.add(keras.layers.Conv2D(192, 3, 2, padding='same', activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(l=0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(l=0.001)))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Softmax())
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = create_model()
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

loss, acc = model.evaluate(valid_dataset, steps=3)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

predictions = np.argmax(model.predict(valid_dataset, steps=1), axis=1)
print("Some predictions: {}".format(predictions[:10]))