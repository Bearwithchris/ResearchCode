# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:06:57 2020

@author: Chris
"""


import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pylab as plt

split = (80, 10, 10)
splits = tfds.Split.TRAIN.subsplit(weighted=split)
(cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs', split=list(splits), with_info=True, as_supervised=True)

# #Data visualisation
# for image, label in cat_train.take(2):
#  plt.figure()
#  plt.imshow(image)

#Preprocessing
IMAGE_SIZE = 100
def pre_process_image(image, label):
 image = tf.cast(image, tf.float32)
 image = image / 255.0
 image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
 return image, label

#Training Data and validation data
TRAIN_BATCH_SIZE = 64
cat_train = cat_train.map(pre_process_image).shuffle(1000).repeat().batch(TRAIN_BATCH_SIZE)
cat_valid = cat_valid.map(pre_process_image).repeat().batch(1000)

#standard classifier model
head = tf.keras.Sequential()
head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
head.add(layers.BatchNormalization())
head.add(layers.Activation('relu'))
head.add(layers.MaxPooling2D(pool_size=(2, 2)))
head.add(layers.Conv2D(32, (3, 3)))
head.add(layers.BatchNormalization())
head.add(layers.Activation('relu'))
head.add(layers.MaxPooling2D(pool_size=(2, 2)))
head.add(layers.Conv2D(64, (3, 3)))
head.add(layers.BatchNormalization())
head.add(layers.Activation('relu'))
head.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Standard classifier end
standard_classifier = tf.keras.Sequential()
standard_classifier.add(layers.Flatten())
standard_classifier.add(layers.BatchNormalization())
standard_classifier.add(layers.Dense(100))
standard_classifier.add(layers.Activation('relu'))
standard_classifier.add(layers.BatchNormalization())
standard_classifier.add(layers.Dense(100))
standard_classifier.add(layers.Activation('relu'))
standard_classifier.add(layers.Dense(1))
standard_classifier.add(layers.Activation('sigmoid'))



# setup a standard classifier model
standard_model = tf.keras.Sequential([
 head, standard_classifier
])


# train the standard classifier model
standard_model.compile(optimizer=tf.keras.optimizers.Adam(),
 loss='binary_crossentropy',
 metrics=['accuracy'])

standard_model.fit(cat_train, steps_per_epoch = 23262//TRAIN_BATCH_SIZE, epochs=10, validation_data=cat_valid,
validation_steps=10, callbacks=callbacks)


# #Global Average pooling model 
# average_pool = tf.keras.Sequential()
# average_pool.add(layers.AveragePooling2D())
# average_pool.add(layers.Flatten())
# average_pool.add(layers.Dense(1, activation='sigmoid'))
# # create the average pooling model
# pool_model = tf.keras.Sequential([
#  head,
#  average_pool])

# # train the standard classifier model
# pool_model.compile(optimizer=tf.keras.optimizers.Adam(),
#  loss='binary_crossentropy',
#  metrics=['accuracy'])

# pool_model.fit(cat_train, steps_per_epoch = 23262//TRAIN_BATCH_SIZE, epochs=10, validation_data=cat_valid,
# validation_steps=10, callbacks=callbacks)
