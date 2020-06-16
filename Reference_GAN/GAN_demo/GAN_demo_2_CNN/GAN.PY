# -*- coding: utf-8 -*-
"""
https://github.com/vincent732/tensorflow-gan-mnist
Created on Sun Jun  7 13:40:07 2020

@author: Chris
"""


import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
# Read the dataset
from tensorflow.examples.tutorials.mnist import input_data
import time
from tqdm import tqdm
import pickle
import imageio
import re

mnist = input_data.read_data_sets("MNIST_data/")

def discriminator(images, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # First convolutional and pool layers
    # This finds 32 different 5 x 5 pixel features
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4
    o = tf.sigmoid(d4)
    
    return o

def generator(z, batch_size, z_dim):
    # From z_dim to 56*56 dimension
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    o = tf.nn.tanh(g4)
    return o


def save_fig(imgs, path, w = 14, h = 14, fig_size=(14, 14), columns = 4, rows = 5):
    assert len(imgs) == columns * rows, "Please check the images"
    fig = plt.figure(figsize=fig_size)
    for i in range(0, columns*rows):
        img = imgs[i]
        fig.add_subplot(rows, columns, i+1)
        if img.ndim == 2:
            plt.gray()
        plt.imshow(img)
    plt.savefig(path)
    plt.close()


def frames_to_gif(directory, output_path="./output.gif"):
    assert os.path.isdir(directory), "Please make sure {} is a folder, and contains images".format(directory)
    images = []
    files = os.listdir(directory)
    ordered_files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))
    for filename in ordered_files:
        path = os.path.join(directory, filename)
        images.append(imageio.imread(path))
    imageio.mimsave(output_path, images)
    
z_dimensions = 100
batch_size = 64
tf.reset_default_graph()

# network: generator
with tf.variable_scope("G"):
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
    # z_placeholder is for feeding input noise to the generator
    Gz = generator(z_placeholder, batch_size, z_dimensions) 
    # Gz holds the generated images

# network: discriminator
with tf.variable_scope("D"):
    x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 
    # x_placeholder is for feeding input images to the discriminator
    Dx = discriminator(x_placeholder) 
    # Dx will hold discriminator prediction probabilities
    # for the real MNIST images
    Dg = discriminator(Gz, reuse=True)
    # Dg will hold discriminator prediction probabilities for generated images
    
eps = 1e-2
# loss for each network
# maximize 1/m * Σlog(Dx) + 1/m * Σ(1-Dg) = min - 1/m * Σlog(Dx) - 1/m * Σ(1-Dg)
D_loss = tf.reduce_mean(-tf.log(Dx + eps) - tf.log(1 - Dg + eps))
G_loss = tf.reduce_mean(-tf.log(Dg + eps))

#Get the variables for different network
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Train the discriminator
d_trainer = tf.train.GradientDescentOptimizer(0.0001).minimize(D_loss, var_list=d_vars)

# Train the generator
g_trainer = tf.train.GradientDescentOptimizer(0.0001).minimize(G_loss, var_list=g_vars)


output_directory = "./MNIST_GAN_results_1000"
# results save folder
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# initial 20 random noise images
target_init_z = np.random.normal(0, 1, size=[20, z_dimensions])
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# trainign stats
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

saver = tf.train.Saver()
epochs = 1000

# Train generator and discriminator together
for i in tqdm(range(1, epochs+1)):
    G_losses = []
    D_losses = []
    for iteration in range(train_set.shape[0] // batch_size):
        with tf.variable_scope("D"):
            real_image_batch = train_set[iteration*batch_size:(iteration+1)*batch_size].reshape([batch_size, 28, 28, 1])
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            
            # Train discriminator
            _, d_loss = sess.run([d_trainer, D_loss],
                                    {x_placeholder: real_image_batch, z_placeholder: z_batch})
            D_losses.append(d_loss)
            
            # Train generator
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            _, g_loss = sess.run([g_trainer, G_loss], feed_dict={z_placeholder: z_batch})
            G_losses.append(g_loss)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    if i % 50 == 0:
        # show generated images
        with tf.variable_scope("G"):
            tf.get_variable_scope().reuse_variables()
            generated_images = generator(z_placeholder, target_init_z.shape[0], z_dimensions)
            images = sess.run(generated_images, {z_placeholder: target_init_z})
            path = os.path.join(output_directory, "epoch_{}.jpg".format(i))
            save_fig(images.squeeze(), path)
        print("Mean Loss for Discriminator:{}, Mean Loss for Generator: {}".format(np.mean(D_losses), np.mean(G_losses)))
saver.save(sess, "./model_1000/model.ckpt")
sess.close()

frames_to_gif("MNIST_GAN_results_1000")


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
z_dimensions = 100


batch_size = 1
with tf.variable_scope("G"):
    tf.get_variable_scope().reuse_variables()
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
    # z_placeholder is for feeding input noise to the generator
    Gz = generator(z_placeholder, batch_size, z_dimensions) 
    # Gz holds the generated images
    
    
saver = tf.train.Saver()

saver.restore(sess, './model_1000/model.ckpt')

test_z = np.random.normal(0, 1, size=[1, z_dimensions])
with tf.variable_scope("G"):
    tf.get_variable_scope().reuse_variables()
    images = sess.run(Gz, {z_placeholder: test_z})
    plt.imshow(images[0].squeeze(), cmap ='gray')
    plt.show()