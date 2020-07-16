# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:58:46 2020

@author: Chris
"""


import tensorflow as tf
import numpy as np
import os 
from tensorflow import keras
import datetime as dt


STORE_PATH="./"
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#Batch Extraction
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)   #Random seed from batch size
    return x_data[idxs,:,:], y_data[idxs]                #Extract the corresponding locations seeded


#Custom Dense Function (To access the weights and bias easier)

class DenseWithInfo(tf.keras.layers.Layer): # inherits the info from the keras class
    def __init__(self,initializer, activation, units=32):
        super(DenseWithInfo,self).__init__() #Initilisa the dense with info of the super class in keras
        
        #override
        self.initializer=initializer
        self.activation=activation
        self.unit=units
        self.logits_sans_bias=None
        self.act_output=None
        
        #override the keras build method 
        def build(self, input_shape):
            self.w=self.add_weight(shape=(input_shape[-1],self.units), initilizer=self.initializer, trainable=True)
            self.b=self.add_weight(shape=(self.units), initilizer=self.initializer, trainable=True)
        
        def call(self, inputs):
            self.logits_sans_bias=tf.matmul(input,self.w)
            logits=self.logits_sans_bias + self.b
            self.act_output=self.activation(logits)
            return self.act_output

#Define model Class            
class Model(object):
    def __init__(self, initialization, activation, num_layers=3, hidden_size=100):
        self.init = initialization
        self.activation = activation
        # num layers does not include the input layer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # create the layers of the model
        self.nn_model = tf.keras.Sequential()
        for i in range(num_layers):
            self.nn_model.add(DenseWithInfo(self.init, self.activation, units=self.hidden_size))
            # don't supply an activation for the final layer - the loss definition will
            # supply softmax activation. This defaults to a linear activation i.e. f(x) = x
            self.nn_model.add(tf.keras.layers.Dense(10, name='output_layer'))
            

    #Feed Forward
    def forward(self, input_images):
        # flatten the input images
        input_images = tf.cast(input_images, tf.float32)
        input_images = tf.reshape(input_images, [-1, 28*28])
        input_images = input_images / 255.0
        logits = self.nn_model(input_images)
        return logits
    
    #Loss Function
    @staticmethod
    def loss(logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    #Log the output
    def log_variables(self, train_writer, step):
        for i in range(self.num_layers):
            # also log the variance of mat mul
            variance = self.calculate_variance(self.nn_model.layers[i].logits_sans_bias)
            with train_writer.as_default():
                tf.summary.histogram(f"mat_mul_hist_{i + 1}", self.nn_model.layers[i].logits_sans_bias, step=step)
                tf.summary.histogram("fc_out_{}".format(i + 1), self.nn_model.layers[i].act_output, step=step)
                tf.summary.scalar("mat_mul_var_{}".format(i + 1), variance, step=step)
    
    #Variances
    @staticmethod
    def calculate_variance(x):
        mean = tf.reduce_mean(x)
        sqr = tf.square(x - mean)
        return tf.reduce_mean(sqr)

def maybe_create_folder_structure(sub_folders):
    for fold in sub_folders:
        if not os.path.isdir(STORE_PATH + fold):
            os.makedirs(STORE_PATH + fold)
        
# def init_pass_through(model, fold):
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     with tf.Session() as sess:
#         sess.run(model.init_op)
#         train_writer = tf.summary.FileWriter(base_path + fold,
#                                              sess.graph)
#         image_batch, label_batch = mnist.train.next_batch(100)
#         summary = sess.run(model.merged, feed_dict={model.input_images: image_batch,
#                                                     model.labels: label_batch})
#         train_writer.add_summary(summary, 0)

# def train_model(model, fold, batch_size, epochs):
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     with tf.Session() as sess:
#         sess.run(model.init_op)
#         train_writer = tf.summary.FileWriter(base_path + fold,
#                                              sess.graph)
#         for i in range(epochs):
#             image_batch, label_batch = mnist.train.next_batch(batch_size)
#             loss, _, acc = sess.run([model.loss, model.optimizer, model.accuracy],
#                                     feed_dict={model.input_images: image_batch,
#                                                     model.labels: label_batch})
#             if i % 50 == 0:
#                 print("Iteration {} of {} - loss: {:.3f}, training accuracy: {:.2f}%".
#                       format(i, epochs, loss, acc*100))
#                 summary = sess.run(model.merged, feed_dict={model.input_images: image_batch,
#                                                     model.labels: label_batch})
#                 train_writer.add_summary(summary, i)
                
def run_training(model: Model, sub_folder: str, iterations: int = 2500, batch_size: int = 32, log_freq: int = 200):
    train_writer = tf.summary.create_file_writer(STORE_PATH + "/" + sub_folder)           #Create Summary file writer (For each respective scenario)
    
    # setup the optimizer
    optimizer = tf.keras.optimizers.Adam()                                                  #Initilise model's optimiser
    for i in range(iterations):                                                             #No. of dfferent "epochs"
        image_batch, label_batch = get_batch(x_train, y_train, batch_size)                  #Extract batch
        image_batch = tf.Variable(image_batch)                                              #Cast tensor variable
        label_batch = tf.cast(tf.Variable(label_batch), tf.int32)                           #Cast to int32
        with tf.GradientTape() as tape:                                                     #Graident calculation and loss determined
                logits = model.forward(image_batch)
                loss = model.loss(logits, label_batch)
                gradients = tape.gradient(loss, model.nn_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.nn_model.trainable_variables))
        if i % log_freq == 0:                                                               #Log Summary progress every 200 iterations 
            max_idxs = tf.argmax(logits, axis=1)
            acc = np.sum(max_idxs.numpy() == label_batch.numpy()) / len(label_batch.numpy())
            print(f"Iter: {i}, loss={loss:.3f}, accuracy={acc * 100:.3f}%")
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=i)                                     #save the lost details
                tf.summary.scalar('accuracy', acc, step=i)                                  #save the accruacy details
            # log the gradients
            model.log_variables(train_writer,i)
                
#Main loop
if __name__ == "__main__":
    #Define the different folders to plot
    sub_folders = ['first_pass_normal', 'first_pass_variance',
    'full_train_normal', 'full_train_variance',
    'full_train_normal_relu', 'full_train_variance_relu',
    'full_train_he_relu']
    
    #Normal initilizer
    initializers = [tf.random_normal_initializer(mean=0.0, stddev=1.0),
                    
    tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='untruncated_normal'),
    tf.random_normal_initializer(mean=0.0, stddev=1.0),
    
    tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='untruncated_normal'),
    tf.random_normal_initializer(mean=0.0, stddev=1.0),
    
    tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='untruncated_normal'),
    tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='untruncated_normal')]
    activations = [tf.sigmoid, tf.sigmoid, tf.sigmoid, tf.sigmoid, tf.nn.relu, tf.nn.relu, tf.nn.relu]
    assert len(sub_folders) == len(initializers) == len(activations)
    maybe_create_folder_structure(sub_folders)
    
    
    for i in range(len(sub_folders)):
        print(f"Scenario: {sub_folders[i]}")
        train_writer = tf.summary.create_file_writer("./" + sub_folders[i])
        model = Model(initializers[i], activations[i])
        run_training(model, sub_folders[i], 32, 1000)