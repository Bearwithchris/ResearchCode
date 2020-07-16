# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:15:26 2020

@author: Chris
"""

from tensorflow import keras
import tensorflow as tf
import datetime as dt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
STORE_PATH="./"

#Batch Extraction
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)   #Random seed from batch size
    return x_data[idxs,:,:], y_data[idxs]                #Extract the corresponding locations seeded

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


#Defining thhe sequence of the model 
model = tf.keras.Sequential([
ConvLayer(tf.nn.relu, 1, 32, [5, 5], [2, 2], [1, 1], [2, 2]),
ConvLayer(tf.nn.relu, 32, 64, [5, 5], [2, 2], [1, 1], [2, 2]),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(300, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal()),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation=None)
])


#Definining the lost function
def loss_fn(logits,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))

#Definining the optimiser
optimizer = tf.keras.optimizers.Adam()

#Defining the training parameters
iterations = 5000
batch_size = 32

#Summary Writter for logging the data
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/MNIST_CNN_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

if __name__=="__main__":
    for i in range(iterations):
        
        #Prepare the batch data 
        batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
        # create tensors
        batch_x = tf.Variable(batch_x) #Create tesnor batch x
        batch_y = tf.Variable(batch_y) #Create tesnor batch y
        batch_y = tf.cast(batch_y, tf.int32) 
        # get the images in the right format (Normalised 0-1)
        batch_x = tf.cast(batch_x, tf.float32)
        batch_x = batch_x / 255.0
        batch_x = tf.reshape(batch_x, (batch_size, 28, 28, 1))
        
        #Gradient descent 
        with tf.GradientTape() as tape:
            logits = model(batch_x) #Tape the mdoel gradient 
            loss = loss_fn(logits, batch_y) #Tape the loss function logits 
        
        #Extract gradient and zip with training variables 
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        #Validate model every 50 iterrations 
        if i % 50 == 0:
            #Evaluate training accuracy 
            max_idxs = tf.argmax(logits, axis=1) #Find the arguement of the high prob prediction
            train_acc = np.sum(max_idxs.numpy() == batch_y.numpy()) / len(batch_y) #Find how many predicted correctly 
            
            #Evaluate test set accuracy
            x_test=tf.cast(tf.Variable(x_test),tf.float32) 
            x_test=tf.reshape(x_test, (x_test.shape[0],28,28,1))
            test_logits = model(x_test, training=False)
            max_idxs = tf.argmax(test_logits, axis=1)
            test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
            print(f"Iter: {i}, loss={loss:.3f}, train accuracy={train_acc * 100:.3f}%, test accuracy={test_acc * 100:.3f}%")
            
            #Log data
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('train_accuracy', train_acc, step=i)
                tf.summary.scalar('test_accuracy', test_acc, step=i)
                
    # determine the test accuracy
    logits = model(x_test, training=False)
    max_idxs = tf.argmax(logits, axis=1)
    acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
    print("Final test accuracy is {:.2f}%".format(acc * 100))