
[上一篇文章](http://blog.csdn.net/u012162613/article/details/45397033)总结了Keras的基本使用方法，相信用过的同学都会觉得不可思议，太简洁了。十多天前，我在github上发现这个框架的时候，关注Keras的人还比较少，这两天无论是github还是微薄，都看到越来越多的人关注和使用Keras。所以这篇文章就简单地再介绍一下Keras的使用,方便各位入门。

主要包括以下三个内容：

- 训练CNN并保存训练好的模型。
- 将CNN用于特征提取，用提取出来的特征训练SVM。
- 可视化CNN卷积层后的特征图。

仍然以Mnist为例，代码中用的Mnist数据到这里下载 
[http://pan.baidu.com/s/1qCdS6](http://pan.baidu.com/s/1qCdS6),本文的代码在我的github上:[dive_into _keras](https://github.com/wepe/MachineLearning/tree/master/DeepLearning%20Tutorials)


----------


###1. 加载数据

数据是图片格式，利用pyhton的PIL模块读取，并转为numpy.array类型。这部分的代码在`data.py`里：


----------


###2. 训练CNN并保存训练好的CNN模型

将上一步加载进来的数据分为训练数据（X_train，30000个样本）和验证数据（X_val，12000个样本），构建CNN模型并训练。训练过程中，每一个epoch得到的val-accuracy都不一样，我们保存达到最好的val-accuracy时的模型，利用Python的cPickle模块保持。（Keras的开发者最近在添加用hdf5保持模型的功能，我试了一下，没用成功，去github发了issue也没人回，估计还没完善，hdf5压缩率会更高，保存下来的文件会更小。）

这部分的代码在`cnn.py`里，运行:

```
python cnn.py
```

在第Epoch 4得到96.45%的validation accuracy,运行完后会得到model.pkl这份文件，保存的就是96.45%对应的模型：

![这里写图片描述](http://img.blog.csdn.net/20150508155724085)


----------


###3.将CNN用于特征提取，用提取出来的特征训练SVM

上一步得到了一个val-accuracy为96.45%的CNN模型，在一些论文中经常会看到用CNN的全连接层的输出作为特征，然后去训练其他分类器。这里我也试了一下，用全连接层的输出作为样本的特征向量，训练SVM。SVM用的是scikit learn里的算法。

这部分代码在`cnn-svm.py`,运行：

```
python cnn-svm.py
```

得到下图的输出，可以看到，cnn-svm的准确率提高到97.89%：

![这里写图片描述](http://img.blog.csdn.net/20150508155806689)


----------


###4.可视化CNN卷积层后的特征图

将卷积层和全连接层后的特征图、特征向量以图片形式展示出来，用到matplotlib这个库。这部分代码在`get_feature_map.py`里。运行：

```
python get_feature_map.py
```

得到全连接层的输出，以及第一个卷积层输出的4个特征图：

![全连接层后的输出](http://img.blog.csdn.net/20150508155842678)

![这里写图片描述](http://img.blog.csdn.net/20150508155724909)

![这里写图片描述](http://img.blog.csdn.net/20150508155810914)

![这里写图片描述](http://img.blog.csdn.net/20150508155833190)

![这里写图片描述](http://img.blog.csdn.net/20150508160043578)


----------
转载请注明出处：http://blog.csdn.net/u012162613/article/details/45581421