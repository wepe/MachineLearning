# -*- coding: utf-8 -*-
"""
Created on Tue Dec 09 21:54:00 2014

@author: wepon

程序说明：
loadData函数
实现的功能是从文件夹中读取所有文件，并将其转化为矩阵返回
如调用loadData('train')，则函数会读取所有的txt文件（'0_0.txt'一直到'1_150.txt'）
并将每个txt文件里的32*32个数字转化为1*1024的矩阵，最终返回大小是m*1024的矩阵
同时返回每个txt文件对应的数字，0或1

sigmoid函数
实现sigmoid函数的功能

gradAscent函数
用梯度下降法计算得到回归系数

classfy函数
根据回归系数对输入的样本进行预测

"""
#!/usr/bin/python
from numpy import *
from os import listdir


def loadData(direction):
    trainfileList=listdir(direction)
    m=len(trainfileList)
    dataArray= zeros((m,1024))
    labelArray= zeros((m,1))
    for i in range(m):
        returnArray=zeros((1,1024))  #每个txt文件形成的特征向量
        filename=trainfileList[i]
        fr=open('%s/%s' %(direction,filename))
        for j in range(32):
            lineStr=fr.readline()
            for k in range(32):
                returnArray[0,32*j+k]=int(lineStr[k])
        dataArray[i,:]=returnArray   #存储特征向量
    
        filename0=filename.split('.')[0]
        label=filename0.split('_')[0]
        labelArray[i]=int(label)     #存储类别
    return dataArray,labelArray
    
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#alpha:步长，maxCycles:迭代次数，可以调整
def gradAscent(dataArray,labelArray,alpha,maxCycles):
    dataMat=mat(dataArray)    #size:m*n
    labelMat=mat(labelArray)      #size:m*1
    m,n=shape(dataMat)
    weigh=ones((n,1)) 
    for i in range(maxCycles):
        h=sigmoid(dataMat*weigh)
        error=labelMat-h    #size:m*1
        weigh=weigh+alpha*dataMat.transpose()*error
    return weigh

def classfy(testdir,weigh):
    dataArray,labelArray=loadData(testdir)
    dataMat=mat(dataArray)
    labelMat=mat(labelArray)
    h=sigmoid(dataMat*weigh)  #size:m*1
    m=len(h)
    error=0.0
    for i in range(m):
        if int(h[i])>0.5:
            print int(labelMat[i]),'is classfied as: 1'
            if int(labelMat[i])!=1:
                error+=1
                print 'error'
        else:
            print int(labelMat[i]),'is classfied as: 0'
            if int(labelMat[i])!=0:
                error+=1
                print 'error'
    print 'error rate is:','%.4f' %(error/m)
                
def digitRecognition(trainDir,testDir,alpha=0.07,maxCycles=10):
    data,label=loadData(trainDir)
    weigh=gradAscent(data,label,alpha,maxCycles)
    classfy(testDir,weigh)
    
        
    
    
    
        
    
        
        
        
        
        