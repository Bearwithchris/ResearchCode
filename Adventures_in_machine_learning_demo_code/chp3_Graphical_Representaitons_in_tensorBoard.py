# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:32:36 2020

@author: Chris
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:16:40 2020
#TensorBoard
@author: Chris
"""


from tensorflow.keras.datasets import mnist
import datetime as dt
import tensorflow as tf
import numpy as np


# create a summary writer for TensorBoard viewing
STORE_PATH="./"
out_file = STORE_PATH + f"/TensorFlow_Visualization_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
train_summary_writer = tf.summary.create_file_writer(out_file)

#Load MNIST DATA Sets
(x_train, y_train) , (x_test , y_test)=mnist.load_data()

#Batch Extraction
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)   #Random seed from batch size
    return x_data[idxs,:,:], y_data[idxs]                #Extract the corresponding locations seeded

#Defining Training Variables 
epochs=10
batch_size=100

#Pre processing: Normalising the data for better processing (Faster)
x_train=x_train/255.0
x_test=x_test/255.0

#Cast to a tesor object
x_test=tf.Variable(x_test)

#Model declaration
W1=tf.Variable(tf.random.normal([784,300],stddev=0.03),name="W1")   #normal inisitalisation 784 nodes inputs ,300 nodes output
b1=tf.Variable(tf.random.normal([300]),name="b1")                   #Normal inisitalisation 300 nodes
W2=tf.Variable(tf.random.normal([300,10],stddev=0.03),name="W2")
b2=tf.Variable(tf.random.normal([10]),name="b2")

#Feed forward
@tf.function() #Convets the Python code into a graph
def nn_model(x_input, labels, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    with tf.name_scope("Hidden") as scope: #Create group name hidden
        hidden_logits = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
        hidden_out = tf.nn.sigmoid(hidden_logits)
    with tf.name_scope("Output") as scope: #Create group name output
        logits = tf.add(tf.matmul(hidden_out, W2), b2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    return logits, hidden_logits, hidden_out, cross_entropy

# # loss function (Back prop)
# def loss_fn(logits,labels):
#     cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
#     return cross_entropy

#Define Optimiser 
optimizer=tf.keras.optimizers.Adam()


genGraph=1

#Training loop
total_batch=int(len(y_train)/batch_size)
for epoch in range(epochs):                                         #First loop for epochs
    avg_loss=0
    for i in range (total_batch):                                   #Second loop for Mini batches
        batch_x, batch_y=get_batch(x_train,y_train, batch_size=batch_size)

        #Create the tensors for inputs
        batch_x=tf.Variable(batch_x)
        batch_y=tf.Variable(batch_y)
        
        #Create the one hot encoding for the cross entropy
        batch_y=tf.one_hot(batch_y,10)
        
        if (genGraph):
            #Tensor Graph
            tf.summary.trace_on(graph=True) # Activate tracing for graph
            logits, _, _, _ = nn_model(batch_x, batch_y, W1, b1, W2, b2)
            with train_summary_writer.as_default(): #Using the summary writer to trace 
                tf.summary.trace_export(name="graph", step=0) #Trace the graph and export into the summary writter
            genGraph=0
        
        

        
        with tf.GradientTape() as tape:                             # Gradient context manager (Keep track on the gradients)
            logits,hidden_logits,hidden_out,loss=nn_model(batch_x,batch_y,W1,b1,W2,b2)
            # loss=loss_fn(logits, batch_y)                           # Compare predicted results from the true results
        gradients=tape.gradient(loss, [W1,b1,W2,b2])                # Retrieve back the gradients from the respective var
        optimizer.apply_gradients(zip(gradients,[W1,b1,W2,b2]))    #Apply gradient descent 
        avg_loss+=loss/total_batch
    
    #Test Model
    test_logits,_,_,_=nn_model(x_test,tf.one_hot(y_test,10),W1,b1,W2,b2)            #One hot encode the y_test also
    max_idxs = tf.argmax(test_logits, axis=1)                                       #Filter only the "best guess"
    test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)                     #Tabulate accuracy i.e. number of correct guesses
    print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
    with train_summary_writer.as_default():                                         
        tf.summary.scalar('loss', avg_loss, step=epoch)                             #log the scalar entries of avg_loss per epoch
        tf.summary.scalar('accuracy', test_acc, step=epoch)                         #log the scalar entries of test_acc per epoch
        tf.summary.histogram("Hidden_logits", hidden_logits, step=epoch)
        tf.summary.histogram("Hidden_output", hidden_out, step=epoch)

print("\nTraining complete!")
    
        
        
        