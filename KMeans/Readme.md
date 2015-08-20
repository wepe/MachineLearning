###KMeans算法，二分KMeans算法

- 封装成class
- 参考sklearn.cluster.KMeans的接口设计


###依赖库
- Numpy
- Matplotlib (可视化结果)


###使用方法

```
import cPickle
from kmeans import KMeans,biKMeans
X,y = cPickle.load(open('data.pkl','r'))
clf = KMeans(n_clusters=10,initCent='random',max_iter=200)
#clf = KMeans(n_clusters=10,initCent=X[0:10],max_iter=200)
#clf = biKMeans(n_clusters=10)
clf.fit(X)
clf.predict(X)

```

###可视化

```

import numpy as np
import matplotlib.pyplot as plt
from kmeans import biKMeans
n_clusters = 10
clf = biKMeans(n_clusters)
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
plt.show()

```

得到下图：
![](http://img.blog.csdn.net/20150820180422017)



