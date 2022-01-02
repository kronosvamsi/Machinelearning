import numpy as np

from numpy import random

import matplotlib.pyplot as plt

a=np.genfromtxt('/content/sample_data/data_regress.txt',dtype=None,encoding=None,delimiter=',')

feature_matrix=np.zeros((201,4))

for i in range(201):
  feature_matrix[i]=[1,a[i,0],a[i,0]*a[i,0],a[i,0]*a[i,0]*a[i,0]]  #creation of feature matrix of size(201,4)

mu1=np.mean(feature_matrix[0:,1])

mu2=np.mean(feature_matrix[0:,2])

mu3=np.mean(feature_matrix[0:,3])

sigma1=np.std(feature_matrix[0:,1])

sigma2=np.std(feature_matrix[0:,2])

sigma3=np.std(feature_matrix[0:,3])

##### normalize feature matrix ###

for i in range(201):
  feature_matrix[i]=[1,a[i,0]-mu1,(a[i,0]**2)-mu2,(a[i,0]**3)-mu3]

norm_matrix=np.zeros((201,4))

for i in range(201):
  norm_matrix[i]=[1,feature_matrix[i,1]/sigma1,feature_matrix[i,2]/sigma2,feature_matrix[i,3]/sigma3]

w=random.rand(4)

cost=np.zeros(201)

alpha=0.05

 ## gradient descent algorithm ###

for k in range(690):    
  for i in range(201):
     cost[i]=(np.sum(norm_matrix[i]*w)-(a[i,1]))**2
     w[0]=w[0]-alpha*((np.sum(norm_matrix[i]*w)-(a[i,1]))*1)
     w[1]=w[1]-alpha*((np.sum(norm_matrix[i]*w)-(a[i,1]))*norm_matrix[i,1])
     w[2]=w[2]-alpha*((np.sum(norm_matrix[i]*w)-(a[i,1]))*norm_matrix[i,2])
     w[3]=w[3]-alpha*((np.sum(norm_matrix[i]*w)-(a[i,1]))*norm_matrix[i,3])
    
