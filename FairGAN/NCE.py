# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:38:16 2020

@author: Chris
"""
import numpy as np

def NCE(gamma,pbiasZ0):
    prefZ0=0.5
    prefZ1=1-prefZ0
    
    pbiasZ1=1-pbiasZ0
    
    bz0=(pbiasZ0/prefZ0)
    bz1=(pbiasZ1/prefZ1)
    
    lhs=(1/(gamma+1))*(((prefZ0)*np.log(1/((gamma*(bz0)+1))))+((prefZ1)*np.log(1/((gamma*bz1+1)))))
    rhs=(gamma/(gamma+1))*(((pbiasZ0)*np.log(gamma*bz0/((gamma*(bz0)+1))))+((pbiasZ1)*np.log(gamma*bz1/((gamma*bz1+1)))))
    
    return lhs+rhs

print(NCE(1.0,0.9))
bias=[0.9,0.8,0.7,0.6,0.5]
bo=[NCE(1.0,x) for x in bias]