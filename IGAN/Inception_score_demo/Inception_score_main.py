# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:55:12 2020

@author: Chris
"""


# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import image_load as il
import matplotlib.pyplot as plt
import numpy as np


def plot(headers,stats,statHeader):
    # Arrange data
    arranged=sorted(stats)
    newLabels=[]
    for arrangedStats in arranged:
        newLabels.append(headers[np.where(stats==arrangedStats)[0][0]])
    
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    y_pos = np.arange(len(newLabels))
    ax.barh(y_pos, arranged, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(newLabels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(statHeader)
    ax.set_title('Data spread')
    
    
# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
    #Image preprocessing=====================================
	# convert from uint8 to float32
	processed = images.astype('float32')
	# pre-process raw images for inception v3 model
	processed = preprocess_input(processed)
    
    #Inception prediction==================================================
	# load inception v3 model
	model = InceptionV3()    
	# predict class probabilities for images
	yhat = model.predict(processed)
    
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

def calculate_cat_inception_score(header,imageList, n_split=10, eps=1E-16):
    isAvgList=[]
    isStdList=[]
    for i in range(len(header)):
        print ("Score and std for "+str(header[i]))
        try:
            is_avg, is_std = calculate_inception_score(il.listToArray(imageList[i]))

            print('score', is_avg, is_std)
            if (is_avg==None or is_std==None):
                isAvgList.append(-1)
                isStdList.append(-1)   
            else:
                isAvgList.append(is_avg)
                isStdList.append(is_std)
        except:
            print ("No info in this category!!!!!!")
            isAvgList.append(-1)
            isStdList.append(-1)
    return (isAvgList,isStdList)
        
    
# pretend to load images
stats,header,imagesList=il.imageStats(20000,True)
# images = ones((50, 299, 299, 3))
print('loaded Images')
# # calculate inception score
# is_avg, is_std = calculate_inception_score(images)
# print('score', is_avg, is_std)
isAvgList,isStdList=calculate_cat_inception_score(header,imagesList,100)
plot(header,isStdList,"std")