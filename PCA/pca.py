# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 19:42:23 2014

@author: wepon

code of PCA Algrithom
"""
import numpy as np
#n features
def pca(dataMat,n):
    #零均值化
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray，若rowvar非0，一列代表一个样本，一行代表一个样本
    
        
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=meanVal*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat
    
    
    
    
    
    
    
    




