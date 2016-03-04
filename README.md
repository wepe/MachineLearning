MachineLearning
====================



一些常见的机器学习算法的实现代码，本人学习过程中做的总结，资历尚浅，如有错误请不吝指出。


##目录介绍

- **DeepLearning Tutorials**

      这个文件夹下包含一些深度学习算法的实现代码，以及具体的应用实例，包含：

	  [dive_into _keras](https://github.com/wepe/MachineLearning/tree/master/DeepLearning%20Tutorials/dive_into_keras) Keras使用进阶。介绍了怎么保存训练好的CNN模型，怎么将CNN用作特征提取，怎么可视化卷积图。[文章链接](http://blog.csdn.net/u012162613/article/details/45581421)
      
      [keras_usage](https://github.com/wepe/MachineLearning/tree/master/DeepLearning%20Tutorials/keras_usage) 介绍了一个简单易用的深度学习框架keras，用经典的Mnist分类问题对该框架的使用进行说明，训练一个CNN，总共不超过30行代码。[文章链接](http://blog.csdn.net/u012162613/article/details/45397033)

      [FaceRecognition_CNN(olivettifaces)](https://github.com/wepe/MachineLearning-Demo/tree/master/DeepLearning%20Tutorials/FaceRecognition_CNN(olivettifaces))
      将卷积神经网络CNN应用于人脸识别的一个demo，人脸数据库采用olivettifaces，CNN模型参考LeNet5，基于python+theano+numpy+PIL实现。详细介绍这个demo的文章：[文章链接](http://blog.csdn.net/u012162613/article/details/43277187)


      [cnn_LeNet](https://github.com/wepe/MachineLearning-Demo/tree/master/DeepLearning%20Tutorials/cnn_LeNet)  CNN卷积神经网络算法的实现，模型为简化版的LeNet，应用于MNIST数据集（手写数字），来自于DeepLearning.net上的一个教程，基于python+theano，我用了中文将原始的代码进行详细的解读，并简单总结了CNN算法，相应的文章发在：[文章链接](http://blog.csdn.net/u012162613/article/details/43225445)

      [mlp](https://github.com/wepe/MachineLearning-Demo/tree/master/DeepLearning%20Tutorials/mlp)  多层感知机算法的实现，代码实现了最简单的三层感知机，并应用于MNIST数据集，来自DeepLearning.net上的一个教程，基于python+theano，我写了一篇文章总结介绍了MLP算法，同时用中文详细解读了原始的代码：[文章链接](http://blog.csdn.net/u012162613/article/details/43221829)

      [Softmax_sgd(or logistic_sgd)](https://github.com/wepe/MachineLearning-Demo/tree/master/DeepLearning%20Tutorials/Softmax_sgd(or%20logistic_sgd)) Softmax回归算法的实现，应用于MNIST数据集，基于Python+theano，来自DeepLearning.net上的一个教程，基于python+theano，我写了一篇文章介绍了Softmax回归算法，同时用中文详细解读了原始的代码：[文章链接](http://blog.csdn.net/u012162613/article/details/43157801)

- **PCA**

      基于python+numpy实现了主成份分析PCA算法，这里详细地介绍了PCA算法，以及代码开发流程：[文章链接](http://blog.csdn.net/u012162613/article/details/42177327)

- **kNN**
      
      基于python+numpy实现了K近邻算法，并将其应用在MNIST数据集上，详细的介绍：[文章链接](http://blog.csdn.net/u012162613/article/details/41768407)

- **logistic regression**

	  - 基于C++以及线性代数库Eigen实现的logistic回归，[代码](https://github.com/wepe/MachineLearning/tree/master/logistic%20regression/use_cpp_and_eigen)

      - 基于python+numpy实现了logistic回归（二类别），详细的介绍：[文章链接](http://blog.csdn.net/u012162613/article/details/41844495)

- **ManifoldLearning**

	[DimensionalityReduction_DataVisualizing](https://github.com/wepe/MachineLearning/tree/master/ManifoldLearning/DimensionalityReduction_DataVisualizing) 运用多种流形学习方法将高维数据降维，并用matplotlib将数据可视化(2维和3维)
     
- **SVM**    

	[libsvm liblinear-usage](https://github.com/wepe/MachineLearning/tree/master/SVM/libsvm%20liblinear-usage) 对使用广泛的libsvm、liblinear的使用方法进行了总结，详细介绍：[文章链接](http://blog.csdn.net/u012162613/article/details/45206813)

- **GMM**

	GMM和k-means作为EM算法的应用，在某种程度有些相似之处，不过GMM明显学习出一些概率密度函数来，结合相关理解写成python版本，详细介绍：[文章链接](http://blog.csdn.net/gugugujiawei/article/details/45583051)

- **DecisionTree**

	Python、Numpy、Matplotlib实现的ID3、C4.5，其中C4.5有待完善，后续加入CART。文章待总结。[代码](https://github.com/wepe/MachineLearning/tree/master/DecisionTree)

- **KMeans**

	介绍了聚类分析中最常用的KMeans算法（及二分KMeans算法），基于NumPy的算法实现，以及基于Matplotlib的聚类过程可视化。[文章链接](http://blog.csdn.net/u012162613/article/details/47811235)

- **NaiveBayes**

	朴素贝叶斯算法的理论推导，以及三种常见模型（多项式模型，高斯模型，伯努利模型）的介绍与编程实现（基于Python，Numpy）。[文章链接](http://blog.csdn.net/u012162613/article/details/48323777)

##Contributor

- [wepon](https://github.com/wepe)
- [Gogary](https://github.com/enjoyhot)
 
