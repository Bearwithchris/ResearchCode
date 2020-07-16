# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:31:54 2020
https://www.tensorflow.org/tutorials/generative/style_transfer
@author: Chris
"""

import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

#Resizing image to max of 512
def load_img(path_to_img):
  max_dim = 512 #Limit to 512 Pixels
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
    
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


#########################################################
#
#   Building the Model
#
#########################################################
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

#Load the VGG model
#This following function builds a VGG19 model that returns a list of intermediate layer outputs:
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

#create the style extractor model and load it
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)


# #Look at the statistics of each layer's output
# for name, output in zip(style_layers, style_outputs):
#   print(name)
#   print("  shape: ", output.numpy().shape)
#   print("  min: ", output.numpy().min())
#   print("  max: ", output.numpy().max())
#   print("  mean: ", output.numpy().mean())
#   print()
  
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


########################################################
# Building the class that returns the style and contet
#########################################################

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    
    #Preprocess the image 
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    #Calculate the Gram_matrix for the image
    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    #Returns a dictionary of dictories that contain layer to matrix mapping 
    return {'content':content_dict, 'style':style_dict}

#Create object for layer extraction
extractor = StyleContentModel(style_layers, content_layers)

#Content layers i.e. the dog image
# results = extractor(tf.constant(content_image))

# print('Styles:')
# for name, output in sorted(results['style'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())
#   print()

# print("Contents:")
# for name, output in sorted(results['content'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())


####################################################
#   Gradient Descent
###################################################

#Style layer i.e. the art piece
style_targets = extractor(style_image)['style']
#Content layer i.e. the dog
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)

#Clip image between 0 to 1
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

#Define the optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

#Defining the loss function
style_weight=1e-2
content_weight=1e4
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    #Sum MSE style loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
    
    style_loss *= style_weight / num_style_layers

    #MSE content loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
    
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    #Extract the layers from the image i.e. the dog
    outputs = extractor(image)
    
    #Calculate the loss
    loss = style_content_loss(outputs)
  
  #Determine the gradient of the loss w.r.t to the image 
  grad = tape.gradient(loss, image)
  
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))
  
# train_step(image)
# train_step(image)
# train_step(image)
# tensor_to_image(image)

#######################################################
#  Main run
#######################################################

import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))


# # #####################################################################
# # # Variational Loss
# # ####################################################################

# # def high_pass_x_y(image):
# #   x_var = image[:,:,1:,:] - image[:,:,:-1,:]
# #   y_var = image[:,1:,:,:] - image[:,:-1,:,:]

# #   return x_var, y_var

# # x_deltas, y_deltas = high_pass_x_y(content_image)

# # plt.figure(figsize=(14,10))
# # plt.subplot(2,2,1)
# # imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

# # plt.subplot(2,2,2)
# # imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

# # x_deltas, y_deltas = high_pass_x_y(image)

# # plt.subplot(2,2,3)
# # imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

# # plt.subplot(2,2,4)
# # imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

# # plt.figure(figsize=(14,10))

# # sobel = tf.image.sobel_edges(content_image)
# # plt.subplot(1,2,1)
# # imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
# # plt.subplot(1,2,2)
# # imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")

# # def total_variation_loss(image):
# #   x_deltas, y_deltas = high_pass_x_y(image)
# #   return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

# # total_variation_loss(image).numpy()

# # tf.image.total_variation(image).numpy()

# # #################################################################
# # #   Rerun Optimisation
# # ################################################################

# # total_variation_weight=30

# # @tf.function()
# # def train_step(image):
# #   with tf.GradientTape() as tape:
# #     outputs = extractor(image)
# #     loss = style_content_loss(outputs)
# #     loss += total_variation_weight*tf.image.total_variation(image)

# #   grad = tape.gradient(loss, image)
# #   opt.apply_gradients([(grad, image)])
# #   image.assign(clip_0_1(image))
  
# # image = tf.Variable(content_image)

# # import time
# # start = time.time()

# # epochs = 10
# # steps_per_epoch = 100

# # step = 0
# # for n in range(epochs):
# #   for m in range(steps_per_epoch):
# #     step += 1
# #     train_step(image)
# #     print(".", end='')
# #   display.clear_output(wait=True)
# #   display.display(tensor_to_image(image))
# #   print("Train step: {}".format(step))

# # end = time.time()
# # print("Total time: {:.1f}".format(end-start))
  
# # file_name = 'stylized-image.png'
# # tensor_to_image(image).save(file_name)

# # try:
# #   from google.colab import files
# # except ImportError:
# #    pass
# # else:
# #   files.download(file_name)