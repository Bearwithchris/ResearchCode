# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:00:55 2020

@author: Chris
"""


import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



dir_anno = "C:/Users/Chris/Documents/Datasets/celebA/"
dir_data = "C:/Users/Chris/Documents/Datasets/celebA/img_align_celeba/"
dir_images = 'C:/Users/Chris/Documents/Datasets/celebA/img_align_celeba_upscale/'

class Images:
    def __init__(self, image, label):
        self.image=image
        self.label=label
        self.score=0
        
    def update_score(self, score):
        self.score=score

    
def get_annotation(txt,countCap):
    imageObjList=np.array([])
    header=0
    count=0
    text=open(dir_anno+txt)
    for line in text:
        line=line.strip("\r\n")
        linList=line.split(" ")
        while ('' in linList):
            linList.remove('')
            
        if header==0:
            headerList=linList
            header=1
        else:
            image=linList[0]
            labelList=[]
            for index in range(len(linList)):
                if (linList[index]=="1"):
                    labelList.append(headerList[index])
            # label=headerList[linList.index("1")]
            im=Images(image,labelList)
            imageObjList=np.append(imageObjList,im)
        # print (linlist)
        
        count+=1
        if (count>countCap):
            break
    return (imageObjList,headerList)

def listStat(imageObjList,headers):
    stats=np.zeros(len(headers))
    for i in imageObjList:
        for j in i.label:       
            indexLabel=headers.index(j)
            stats[indexLabel]+=1
   
    #Re order data=================
    arranged=sorted(stats)
    newLabels=[]
    for arrangedStats in arranged:
        newLabels.append(headers[np.where(stats==arrangedStats)[0][0]])
    
    headers=newLabels
    stats=arranged
    del headers[0]
    del stats[0]
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    y_pos = np.arange(len(headers))
    ax.barh(y_pos, stats, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(headers)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('count')
    ax.set_title('Data spread')
    
    return stats

def loadImges(listNames):
    count=0
    imagelist=[]
    for i in listNames:
        images=Image.open(dir_images+listNames)
        images=np.asarray(images)
        imagelist.append(images)

    return np.array(imagelist)

def loadCatImages(attr,header):
    # catList=[[] for i in range(len(header))]
    # for i in attr:
    #     images=Image.open(dir_images+i.image)
    #     images=np.asarray(images)
    #     for j in i.label:
    #         catList[(header.index(j))].append(images)
    # for i in range(len(catList)):
    #     catList[i]=np.asarray(catList[i])
    
    #Less memory intensive (SAVE name of image to be loaded)
    catList=[[] for i in range(len(header))]
    for i in attr:
        for j in i.label:
            catList[(header.index(j))].append(i.image)
        
    return (catList)

def listToArray(selList):
    arrayList=[]
    for i in selList:
        images=Image.open(dir_images+i)
        images=np.asarray(images)
        arrayList.append(images)
    return(np.asarray(arrayList))
    
    
        
        

def imageStats(countCap,cat=False):
    txt="list_attr_celeba.txt"
    attr, header= get_annotation(txt,countCap)
    stats=listStat(attr,header)
    if (cat):
        imagesList=loadCatImages(attr,header)
    else:
        imagesList=loadImges(attr)
        
    return stats,header,imagesList

    


# stats,header,imagesList=imageStats(10,True)
# array=listToArray(imagesList[1])


