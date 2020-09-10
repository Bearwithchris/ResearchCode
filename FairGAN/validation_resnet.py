# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:20:33 2020

@author: Chris
"""
import tensorflow as tf
import os
import fnmatch
import data
import numpy as np
import datetime as dt
from resnet import resnet18

checkpoints="./training_checkpoints"
# checkpoints="./restnet18_bias_point9_training_checkpoints"

test_dir1 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/Male'
test_dir2 = 'H:/Datasets/celebA_Male_female/669126_1178231_bundle_archive/Dataset/Test/female'

STORE_PATH="./"
out_file_Test = STORE_PATH + f"/TEST_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
train_summary_writer_Test = tf.summary.create_file_writer(out_file_Test)

def create_model():  
    # base_model=tf.keras.applications.InceptionV3(
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=None,
    #     input_shape=[227,227,3],
    #     pooling=None,
    #     classes=2
    # )
    
    # # tf.keras.utils.plot_model(base_model , to_file='model_base.png', show_shapes=True, dpi=64) #Added to visualise model
    # base_model.trainable=False
    
    # model = tf.keras.Sequential()
    # model.add(base_model)
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(2))
    # model.add(tf.keras.layers.Softmax())
    model=resnet18.make_resnet_18().call(isTrain=False)
    model.trainable=False
    
    optimizer=tf.keras.optimizers.Adam(1e-4)
    
    return model,optimizer

def load_checkpoints(directory):
    files = fnmatch.filter(os.listdir(checkpoints), "*.index")
    files =[i.strip('.index') for i in files]
    return files
    
def load_model(checkpoint_path,model):
    #Optimisers
    
    
    #Models
    # model,optimizer= create_model()
    checkpoint_dir = os.path.join(checkpoints, checkpoint_path)
    checkpoint = tf.train.Checkpoint(main_model=model, optimizer=optimizer)
    
    checkpoint.restore(checkpoint_dir)
    
    return model

def cross_entropy_loss(expected_labels, predicted_labels):
    cross_entropy =  tf.keras.losses.SparseCategoricalCrossentropy()
    loss = cross_entropy(expected_labels, predicted_labels)
    return loss 

def sort(listDir):
    numerics=[int(x.strip('ckpt-')) for x in listDir]
    placeHolder=["blank" for i in range(len(numerics))]
    for i in range(len(numerics)):
        placeHolder[numerics[i]-1]=listDir[i]
    return placeHolder

images_labels_concat_shuffled,images_concat_shuffled=data.datasets(test_dir1,test_dir2,13000,bias=0.6)

EPOCH=15
BATCH_SIZE=64
batch_num=len(images_labels_concat_shuffled)/BATCH_SIZE

images_Test = tf.data.Dataset.from_tensor_slices(images_concat_shuffled).batch(BATCH_SIZE)
images_Test_Labels = tf.data.Dataset.from_tensor_slices(images_labels_concat_shuffled).batch(BATCH_SIZE)
    

if __name__=="__main__":
    listDir=load_checkpoints(checkpoints)
    listDir=sort(listDir)
    model,optimizer= create_model()
    
    
    epoch=0
    for i in listDir:
        model=load_model(i,model)
        losses=[]
        acc=[]
        
        for image_batch,label_batch in zip(images_Test,images_Test_Labels):
        
            # Test
            predicted_labels_test=model(image_batch)
            
            #Loss
            loss_Test=cross_entropy_loss(label_batch,predicted_labels_test)
            losses.append(loss_Test.numpy())
            
            #Accuracy
            max_idxs = tf.argmax(predicted_labels_test, axis=1)
            train_acc = np.sum(max_idxs.numpy() == label_batch) / len(label_batch)
            acc.append(train_acc)
            
        print ("Epoch: "+str(epoch)+" Mean loss: "+ str(np.mean(np.array(losses)))+" Mean Acc:"+ str(np.mean(np.array(acc))))    
       
            
        with train_summary_writer_Test.as_default():                                         
            tf.summary.scalar('loss', np.mean(np.array(losses)), step=epoch)  
            tf.summary.scalar('acc', np.mean(np.array(acc)), step=epoch)   
        epoch+=1