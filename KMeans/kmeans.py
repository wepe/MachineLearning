#coding=utf-8

'''
@author: wepon, http://2hwp.com
Reference:
            Book: <<Machine Learning in Action>>
            Software: sklearn.cluster.KMeans

'''
import numpy as np

class KMeans(object):
    """
    - 参数
        n_clusters:
            聚类个数，即k
        initCent:
            质心初始化方式，可选"random"或指定一个具体的array,默认random，即随机初始化
        max_iter:
            最大迭代次数
    """
    def __init__(self,n_clusters=5,initCent='random',max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None   
        self.sse = None 
    
    #计算两点的欧式距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)
        
    #随机选取k个质心,必须在数据集的边界内
    def _randCent(self, X, k):
        n = X.shape[1]        #特征维数
        centroids = np.empty((k,n))  #k*n的矩阵，用于存储质心
        for j in range(n):           #产生k个质心，一维一维地随机初始化
            minJ = min(X[:,j]) 
            rangeJ = float(max(X[:,j]) - minJ)
            centroids[:,j] = (minJ + rangeJ * np.random.rand(k,1)).flatten()
        return centroids
        
    def fit(self, X):
        #类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        m = X.shape[0]#m代表样本数量
        self.clusterAssment = np.empty((m,2))#m*2的矩阵，第一列存储样本点所属的族的索引值，
                                               #第二列存储该点与所属族的质心的平方误差
        if self.initCent == 'random':
             self.centroids = self._randCent(X, self.n_clusters)
        
        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):#将每个样本点分配到离它最近的质心所属的族
                minDist = np.inf; minIndex = -1
                for j in range(self.n_clusters):
                    distJI = self._distEclud(self.centroids[j,:],X[i,:])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                if self.clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i,:] = minIndex,minDist**2
                    
            if not clusterChanged:#若所有样本点所属的族都不改变,则已收敛，结束迭代
                break   
            for i in range(self.n_clusters):#更新质心，即将每个族中的点的均值作为质心
                ptsInClust = X[np.nonzero(self.clusterAssment[:,0]==i)[0]]#取出属于第i个族的所有点
                self.centroids[i,:] = np.mean(ptsInClust, axis=0)
        
        self.labels = self.clusterAssment[:,0]
        self.sse = sum(self.clusterAssment[:,1])

    
    def predict(self,X):#根据聚类结果，预测新输入数据所属的族
        #类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        m = X.shape[0]#m代表样本数量
        preds = np.empty((m,))
        for i in range(m):#将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j,:],X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds
    
    
class biKMeans(object):
    def __init__(self,n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusterAssment = None
        self.labels = None
        self.sse = None
        
    
    #计算两点的欧式距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)
        
    def fit(self,X):
        m = X.shape[0]
        self.clusterAssment = np.zeros((m,2))
        centroid0 = np.mean(X, axis=0).tolist()
        centList =[centroid0]
        for j in range(m):#计算每个样本点与质心之间初始的平方误差
            self.clusterAssment[j,1] = self._distEclud(np.asarray(centroid0), X[j,:])**2
        
        while (len(centList) < self.n_clusters):
            lowestSSE = np.inf
            for i in range(len(centList)):#尝试划分每一族,选取使得误差最小的那个族进行划分
                ptsInCurrCluster = X[np.nonzero(self.clusterAssment[:,0]==i)[0],:]
                clf = KMeans(n_clusters=2)
                clf.fit(ptsInCurrCluster)
                centroidMat, splitClustAss = clf.centroids, clf.clusterAssment#划分该族后，所得到的质心、分配结果及误差矩阵
                sseSplit = sum(splitClustAss[:,1])
                sseNotSplit = sum(self.clusterAssment[np.nonzero(self.clusterAssment[:,0]!=i)[0],1])
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            #该族被划分成两个子族后,其中一个子族的索引变为原族的索引，另一个子族的索引变为len(centList),然后存入centList
            bestClustAss[np.nonzero(bestClustAss[:,0] == 1)[0],0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:,0] == 0)[0],0] = bestCentToSplit
            centList[bestCentToSplit] = bestNewCents[0,:].tolist()
            centList.append(bestNewCents[1,:].tolist())
            self.clusterAssment[np.nonzero(self.clusterAssment[:,0] == bestCentToSplit)[0],:]= bestClustAss 
                   
        self.labels = self.clusterAssment[:,0] 
        self.sse = sum(self.clusterAssment[:,1])
        self.centroids = np.asarray(centList)
                                
    def predict(self,X):#根据聚类结果，预测新输入数据所属的族
        #类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        m = X.shape[0]#m代表样本数量
        preds = np.empty((m,))
        for i in range(m):#将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j,:],X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds
