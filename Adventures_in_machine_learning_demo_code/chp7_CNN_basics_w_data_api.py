# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:16:25 2020

@author: Chris
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


#===========================================================================================================
#           From previous chapter
#==========================================================================================================
# #Batch Extraction
# def get_batch(x_data, y_data, batch_size):
#     idxs = np.random.randint(0, len(y_data), batch_size)   #Random seed from batch size
#     return x_data[idxs,:,:], y_data[idxs]                #Extract the corresponding locations seeded

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, activation, input_channels, output_channels, window_size, pool_size, filt_stride, pool_stride, initializer=tf.keras.initializers.he_normal()):
        super(ConvLayer, self).__init__() #Inherit the convLayer from Keras
        #Initialise 
        self.initializer = initializer
        self.activation = activation
        self.input_channels = input_channels
        self.output_channels = output_channels
    
        self.window_size = window_size
        self.pool_size = pool_size
        self.filt_stride = filt_stride
        self.pool_stride = pool_stride
        
        #Intilialise filter size 
        self.w = self.add_weight(shape=(window_size[0], window_size[1], input_channels, output_channels), initializer=self.initializer, trainable=True)
        self.b = self.add_weight(shape=(output_channels,), initializer=tf.zeros_initializer, trainable=True)
        
        
    def call(self, inputs):
        #Convolude the filter over the layer
        filt_stride = [1, self.filt_stride[0], self.filt_stride[1], 1] #Definining the filter dimensions [1 x stride x stride x 1] assume 1 channel 
        out_layer = tf.nn.conv2d(inputs, self.w, filt_stride, padding='SAME') #Convolution 2D the input with weight *Note that this is diff from Keras.layer.conv2d
        
        # add the bias
        out_layer += self.b # Add bias 
        out_layer = self.activation(out_layer) #Activate the output layer
        
        #Intitialise the pooling layer and pool
        pool_shape = [1, self.pool_size[0], self.pool_size[1], 1] 
        pool_strides = [1, self.pool_stride[0], self.pool_stride[1], 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=pool_shape, strides=pool_strides, padding='SAME')
        return out_layer


#Definining the lost function
def loss_fn(logits,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))

#==========================================================================================================


# load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# create the training datasets
dx_train = tf.data.Dataset.from_tensor_slices(x_train).map(lambda x: tf.cast(x, tf.float32) / 255.0)
dx_train = dx_train.map(lambda x: tf.reshape(x, (28, 28, 1)))
dy_train = tf.data.Dataset.from_tensor_slices(y_train).map(lambda x: tf.cast(x, tf.int32))
# zip the x and y training data together and shuffle, batch etc. (New addition using the API)
train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).repeat().shuffle(500).batch(32)



# do the same operations for the test set (New addition using the API)
dx_test= tf.data.Dataset.from_tensor_slices(x_test).map(lambda x: tf.cast(x, tf.float32) / 255.0)
dx_test = dx_test.map(lambda x: tf.reshape(x, (28, 28, 1)))
dy_test = tf.data.Dataset.from_tensor_slices(y_test).map(lambda x: tf.cast(x, tf.int32))
test_dataset = tf.data.Dataset.zip((dx_test, dy_test)).repeat().shuffle(1000).batch(500)

# create iterators
train_iter = iter(train_dataset)
test_iter = iter(test_dataset)


# create the neural network model
model = tf.keras.Sequential([
ConvLayer(tf.nn.relu, 1, 32, [5, 5], [2, 2], [1, 1], [2, 2]),
ConvLayer(tf.nn.relu, 32, 64, [5, 5], [2, 2], [1, 1], [2, 2]),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(300, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal()),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation=None)
])
# run the training
iterations = 1000
optimizer = tf.keras.optimizers.Adam()

for i in range(iterations):
    batch_x, batch_y = next(train_iter)                                             #Replace get_batch function
    with tf.GradientTape() as tape:
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if i % 50 == 0:
         max_idxs = tf.argmax(logits, axis=1)
         train_acc = np.sum(max_idxs.numpy() == batch_y.numpy()) / len(batch_y)
         # get the test data
         x_test, y_test = next(test_iter)
         test_logits = model(x_test, training=False)
         max_idxs = tf.argmax(test_logits, axis=1)
         test_acc = np.sum(max_idxs.numpy() == y_test.numpy()) / len(y_test)
         print(f"Iter: {i}, loss={loss:.3f}, "f"train accuracy={train_acc * 100:.3f}%, test accuracy={test_acc * 100:.3f}%")
     