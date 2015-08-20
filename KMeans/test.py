#coding=utf-8
import cPickle
import matplotlib.pyplot as plt
import numpy as np
from kmeans import KMeans,biKMeans
    
if __name__ == "__main__":
    #加载数据
    X,y = cPickle.load(open('data.pkl','r'))

    #依次画出迭代1次、2次、3次...的图
    for max_iter in range(6):
        #设置参数
        n_clusters = 10
        initCent = X[50:60] #将初始质心初始化为X[50:60]
        #训练模型
        clf = KMeans(n_clusters,initCent,max_iter)
        clf.fit(X)
        cents = clf.centroids
        labels = clf.labels
        sse = clf.sse
        #画出聚类结果，每一类用一种颜色
        colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
        for i in range(n_clusters):
            index = np.nonzero(labels==i)[0]
            x0 = X[index,0]
            x1 = X[index,1]
            y_i = y[index]
            for j in range(len(x0)):
                plt.text(x0[j],x1[j],str(int(y_i[j])),color=colors[i],\
                         fontdict={'weight': 'bold', 'size': 9})
            plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=12)
        plt.title("SSE={:.2f}".format(sse))
        plt.axis([-30,30,-30,30])
        #plt.savefig("{}.png".format(max_iter))
        #plt.close()
        plt.show()
