# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:04:33 2015

@author: wepon
"""

import numpy as np

class DecisionTree:
    """决策树使用方法：
    
        - 生成实例： clf = DecisionTrees(). 参数mode可选，ID3或C4.5，默认C4.5
        
        - 训练，调用fit方法： clf.fit(X,y).  X,y均为np.ndarray类型
                            
        - 预测，调用predict方法： clf.predict(X). X为np.ndarray类型
                                 
        - 可视化决策树，调用showTree方法 
    
    """
    def __init__(self,mode='C4.5'):
        self._tree = None
        
        if mode == 'C4.5' or mode == 'ID3':
            self._mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')
        
            
    
    def _calcEntropy(self,y):
        """
        函数功能：计算熵
        参数y：数据集的标签
        """
        num = y.shape[0]
        #统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys(): labelCounts[label] = 0
            labelCounts[label] += 1
        #计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/num
            entropy -= prob * np.log2(prob)
        return entropy
    
    
    
    def _splitDataSet(self,X,y,index,value):
        """
        函数功能：返回数据集中特征下标为index，特征值等于value的子数据集
        """
        ret = []
        featVec = X[:,index]
        X = X[:,[i for i in range(X.shape[1]) if i!=index]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                ret.append(i)
        return X[ret,:],y[ret]
    
    
    def _chooseBestFeatureToSplit_ID3(self,X,y):
        """ID3
        函数功能：对输入的数据集，选择最佳分割特征
        参数dataSet：数据集，最后一列为label
        主要变量说明：
                numFeatures：特征个数
                oldEntropy：原始数据集的熵
                newEntropy：按某个特征分割数据集后的熵
                infoGain：信息增益
                bestInfoGain：记录最大的信息增益
                bestFeatureIndex：信息增益最大时，所选择的分割特征的下标
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        #对每个特征都计算一下infoGain，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):        
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            #进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)  
            #计算信息增益，根据信息增益选择最佳分割特征
            infoGain = oldEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex
        
    def _chooseBestFeatureToSplit_C45(self,X,y):
        """C4.5
            ID3算法计算的是信息增益，C4.5算法计算的是信息增益比，对上面ID3版本的函数稍作修改即可
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestGainRatio = 0.0
        bestFeatureIndex = -1
        #对每个特征都计算一下gainRatio=infoGain/splitInformation
        for i in range(numFeatures):        
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            splitInformation = 0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            #进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)  
                splitInformation -= prob * np.log2(prob)
            #计算信息增益比，根据信息增益比选择最佳分割特征
            #splitInformation若为0，说明该特征的所有值都是相同的，显然不能作为分割特征
            if splitInformation==0.0:
                pass
            else:
                infoGain = oldEntropy - newEntropy
                gainRatio = infoGain/splitInformation
                if(gainRatio > bestGainRatio):
                    bestGainRatio = gainRatio
                    bestFeatureIndex = i
        return bestFeatureIndex
    
    
    
    def _majorityCnt(self,labelList):
        """
        函数功能：返回labelList中出现次数最多的label
        """
        labelCount={}
        for vote in labelList:
            if vote not in labelCount.keys(): labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(),key=lambda x:x[1], reverse=True)
        return sortedClassCount[0][0]
    
    
    
    def _createTree(self,X,y,featureIndex):
        """建立决策树
        featureIndex，类型是元组，它记录了X中的特征在原始数据中对应的下标。
        """
        labelList = list(y)
        #所有label都相同的话，则停止分割，返回该label
        if labelList.count(labelList[0]) == len(labelList): 
            return labelList[0]
        #没有特征可分割时，停止分割，返回出现次数最多的label
        if len(featureIndex) == 0:
            return self._majorityCnt(labelList)
        
        #可以继续分割的话，确定最佳分割特征
        if self._mode == 'C4.5':
            bestFeatIndex = self._chooseBestFeatureToSplit_C45(X,y)
        elif self._mode == 'ID3':
            bestFeatIndex = self._chooseBestFeatureToSplit_ID3(X,y)
            
        bestFeatStr = featureIndex[bestFeatIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex = tuple(featureIndex)
        #用字典存储决策树。最佳分割特征作为key，而对应的键值仍然是一棵树（仍然用字典存储）
        myTree = {bestFeatStr:{}}
        featValues = X[:,bestFeatIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            #对每个value递归地创建树
            sub_X,sub_y = self._splitDataSet(X,y, bestFeatIndex, value)
            myTree[bestFeatStr][value] = self._createTree(sub_X,sub_y,featureIndex)
        return myTree  
    
    def fit(self,X,y):
        #类型检查
        if isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            pass
        else: 
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")
        
        featureIndex = tuple(['x'+str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X,y,featureIndex)
        return self  #allow chaining: clf.fit().predict()

    

    def predict(self,X):
        if self._tree==None:
            raise NotFittedError("Estimator not fitted, call `fit` first")
        
        #类型检查
        if isinstance(X,np.ndarray): 
            pass
        else: 
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        def _classify(tree,sample):
            """
            用训练好的决策树对输入数据分类 
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本（sample）分类
            To Do: 多个sample的预测怎样并行化？
            """
            featIndex = tree.keys()[0]
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])]
            valueOfkey = secondDict[key]
            if isinstance(valueOfkey, dict): 
                label = _classify(valueOfkey,sample)
            else: label = valueOfkey
            return label
            
        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:   
            results = []
            for i in range(X.shape[0]):
                results.append(_classify(self._tree,X[i]))
            return np.array(results)
        
    def show(self):
        if self._tree==None:
            raise NotFittedError("Estimator not fitted, call `fit` first")
        
        #plot the tree using matplotlib
        import treePlotter
        treePlotter.createPlot(self._tree)

     
class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting
    
    """
    pass