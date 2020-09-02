# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:05:06 2020

@author: Chris
"""

import tensorflow as tf
from residual_block import make_basic_block_layer

NUM_CLASSES = 2

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()
        
        self.gen_input_shape=[227,227,3]
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.dropout= tf.keras.layers.Dropout(0.2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self,isTrain=True, training=None, mask=None):
        inp=tf.keras.layers.Input(shape=self.gen_input_shape)
        x = self.conv1(inp)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        if isTrain:
            x= self.dropout(x)
        output = self.fc(x)
        
        x=tf.keras.Model(inputs=inp,outputs=output)
        tf.keras.utils.plot_model(x , to_file='resnet.png', show_shapes=True, dpi=64) #Added to visualise model
        

        return x


def make_resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])

model=make_resnet_18().call()