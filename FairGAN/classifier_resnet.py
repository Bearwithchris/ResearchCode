# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:56:18 2020

@author: Chris
"""

import tensorflow as tf
import data
import numpy as np
import datetime as dt
import os 
from resnet import resnet18

# create a summary writer for TensorBoard viewing
STORE_PATH="./"
out_file = STORE_PATH + f"/Train_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
train_summary_writer = tf.summary.create_file_writer(out_file)

# out_file_Test = STORE_PATH + f"/TEST_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
# train_summary_writer_Test = tf.summary.create_file_writer(out_file_Test)

train_dir1 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Train/Male'
train_dir2 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Train/female'
# images_concat_shuffled,images_labels_concat_shuffled=data.prep_dataset(train_dir1,train_dir2)

images_labels_concat_shuffled,images_concat_shuffled=data.datasets(train_dir1,train_dir2,13000,bias=0.9)




# test_dir1 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/Male'
# test_dir2 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/female'

model=resnet18.make_resnet_18().call()
tf.keras.utils.plot_model(model , to_file='resnet.png', show_shapes=True, dpi=64) #Added to visualise model

optimizer=tf.keras.optimizers.RMSprop(1e-4)

EPOCH=15
BATCH_SIZE=64
batch_num=len(images_labels_concat_shuffled)/BATCH_SIZE

def cross_entropy_loss(expected_labels, predicted_labels):
    cross_entropy =  tf.keras.losses.SparseCategoricalCrossentropy()
    loss = cross_entropy(expected_labels, predicted_labels)
    return loss 
    
   
@tf.function
def train_step(images,real_labels):
    # for epoch in range(EPOCH):
    #     for batch in range (bath_num):
    with tf.GradientTape() as tape:
        predicted_labels=model(images)
        loss=cross_entropy_loss(real_labels,predicted_labels)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    
    max_idxs = tf.argmax(predicted_labels, axis=1)
    # train_acc = np.sum(max_idxs.numpy() == real_labels) / len(real_labels) 
    return loss,max_idxs
    
def train(dataset,labels):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model_optimizer=optimizer,
                                 main_model=model)
    for epoch in range(EPOCH):  
        # dataset=iter(dataset)
        # labels=iter(labels)
        count=0
        losses_Train=[]
        acc_Train=[]
        for image_batch,label_batch in zip(dataset,labels):
            count+=1
            # dataset=iter(dataset)
            # labels=iter(labels)
            # image_batch=next(dataset)
            # label_batch=next(labels)
            loss,max_idxs=train_step(image_batch,label_batch)
            losses_Train.append(loss)
            
            #Accruacy
            train_acc = np.sum(max_idxs.numpy() == label_batch) / len(label_batch)
            acc_Train.append(train_acc)
            
            # print ("Epoch: "+str(epoch)+" batch: "+str(count)+" Loss:"+str(loss))
        print ("Epoch: "+str(epoch)+" Mean loss: "+ str(np.mean(np.array(losses_Train)))+" Mean Acc:"+ str(np.mean(np.array(acc_Train))))    
        with train_summary_writer.as_default():                                         
            tf.summary.scalar('loss', np.mean(np.array(losses_Train)), step=epoch)  
            tf.summary.scalar('acc', np.mean(np.array(acc_Train)), step=epoch)  
  
        # #Test
        # images_Test,images_Test_Labels=data.prep_dataset_Test(test_dir1,test_dir2)
        # predicted_labels_test=model(images_Test)
        # loss_Test=cross_entropy_loss(images_Test_Labels,predicted_labels_test)
        # with train_summary_writer_Test.as_default():                                         
        #     tf.summary.scalar('loss',loss_Test, step=epoch)  
        
        

        
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    

if __name__=="__main__":
    images_concat_shuffled = tf.data.Dataset.from_tensor_slices(images_concat_shuffled).batch(BATCH_SIZE)
    images_labels_concat_shuffled = tf.data.Dataset.from_tensor_slices(images_labels_concat_shuffled).batch(BATCH_SIZE)
    train(images_concat_shuffled,images_labels_concat_shuffled)

    
            