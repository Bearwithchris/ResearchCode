# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:56:09 2020

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


#Creating an object of the model
class Model(object):
    #Initilise model
    def __init__(self,activation,num_layers=6,hidden_size=10):
        self.num_layers = num_layers
        self.nn_model = tf.keras.Sequential()
        
        for i in range(num_layers):
            self.nn_model.add(tf.keras.layers.Dense(hidden_size,activation=activation,name=f'layer{i+1}'))
        self.nn_model.add(tf.keras.layers.Dense(10,name='output_layer'))
    
    #Forward pass
    @tf.function()
    def forward(self,input_images):
        input_images=tf.cast(input_images,tf.float32)
        input_images=tf.reshape(input_images,[-1,28*28])/255.0
        logits=self.nn_model(input_images)
        return logits

    #Loss function
    @staticmethod        
    def loss(logits,labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    
    #Plot gradient details
    def log_gradients(self,gradients,train_writer,step):
        
        assert len(gradients) == len(self.nn_model.trainable_variables)
        
        for i in range(len(gradients)):
            if 'kernel' in self.nn_model.trainable_variables[i].name:                       #Expliciting extracting the weights with the term "kernel"
                with train_writer.as_default():
                    tf.summary.scalar(f"mean_{int((i - 1) / 2)}", tf.reduce_mean(tf.abs(gradients[i])), step=step)
                    tf.summary.histogram(f"histogram_{int((i - 1) / 2)}", gradients[i], step=step)
                    tf.summary.histogram(f"hist_weights_{int((i - 1) / 2)}", self.nn_model.trainable_variables[i],step=step)
     
    #Plot graph
    def plot_computational_graph(self, train_writer, x_batch):
        tf.summary.trace_on(graph=True)
        self.forward(x_batch)
        with train_writer.as_default():
            tf.summary.trace_export(name="graph", step=0)
            
def run_training(model: Model, sub_folder: str, iterations: int = 2500, batch_size: int = 32, log_freq: int = 200):
    train_writer = tf.summary.create_file_writer(STORE_PATH + "/" + sub_folder)             #Create Summary file writer (For each respective scenario)
    model.plot_computational_graph(train_writer, x_train[:batch_size, :, :])                #Plot Model Graph
    
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
            model.log_gradients(gradients, train_writer, i)
                   
scenarios = ["sigmoid", "relu", "leaky_relu"]
act_funcs = [tf.sigmoid, tf.nn.relu, tf.nn.leaky_relu]
assert len(scenarios) == len(act_funcs)
# collect the training data
for i in range(len(scenarios)):
    print(f"Running scenario: {scenarios[i]}")
    model = Model(act_funcs[i], 6, 10)
    run_training(model, scenarios[i])