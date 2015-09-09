
###github不支持Latex，公式显示不了，请查看[博文](http://blog.csdn.net/u012162613/article/details/48323777)


-------


朴素贝叶斯（Naive Bayes）是一种简单的分类算法，它的经典应用案例为人所熟知：文本分类（如垃圾邮件过滤）。很多教材都从这些案例出发，本文就不重复这些内容了，而把重点放在理论推导（其实很浅显，别被“理论”吓到），三种常用模型及其编码实现（Python）。

如果你对理论推导过程不感兴趣，可以直接逃到三种常用模型及编码实现部分，但我建议你还是看看理论基础部分。

另外，本文的所有代码都可以在[这里获取]()

#1. 朴素贝叶斯的理论基础

>朴素贝叶斯算法是基于贝叶斯定理与特征条件独立假设的分类方法。

这里提到的**贝叶斯定理**、**特征条件独立假设**就是朴素贝叶斯的两个重要的理论基础。

##1.1 贝叶斯定理

先看什么是**条件概率**。

P(A|B)表示事件B已经发生的前提下，事件A发生的概率，叫做事件B发生下事件A的条件概率。其基本求解公式为：$P(A|B)=\frac{P(AB)}{P(B)}$

贝叶斯定理便是基于条件概率，通过P(A|B)来求P(B|A)：

$P(B|A)=\frac{P(A|B)P(B)}{P(A)}$

顺便提一下，上式中的分母P(A),可以根据全概率公式分解为：

$P(A)=\sum_{i=1}^{n}P(B_{i})P(A|B_{i})$


##1.2 特征条件独立假设

这一部分开始朴素贝叶斯的理论推导，从中你会深刻地理解什么是特征条件独立假设。

给定训练数据集（X,Y），其中每个样本x都包括n维特征，即$x=({x_{1},x_{2},x_{3},...,x_{n}})$，类标记集合含有k种类别，即$y=({y_{1},y_{2},...,y_{k}})$。

如果现在来了一个新样本x，我们要怎么判断它的类别？从概率的角度来看，这个问题就是给定x，它属于哪个类别的概率最大。那么问题就转化为求解$P(y_{1}|x),P(y_{2}|x),...,P(y_{k}|x)$中最大的那个，即求后验概率最大的输出：$argmax_{y_{k}}   P(y_{k}|x)$

那$P(y_{k}|x)$怎么求解？答案就是贝叶斯定理：

$P(y_{k}|x)=\frac{P(x|y_{k})P(y_{k})}{P(x)}$

根据全概率公式，可以进一步地分解上式中的分母：

$P(y_{k}|x)=\frac{P(x|y_{k})P(y_{k})}{\sum_{k}P(x|y_{k})P(y_{k})}$     **【公式1】**

>这里休息两分钟

先不管分母，分子中的$P(y_{k})$是先验概率，根据训练集就可以简单地计算出来。

而条件概率$P(x|y_{k})=P(x_{1},x_{2},...,x_{n}|y_{k})$，它的参数规模是指数数量级别的，假设第i维特征$x_{i}$可取值的个数有$S_{i}$个，类别取值个数为k个，那么参数个数为：$k\prod_{i=1}^{n}S_{i}$

这显然不可行。**针对这个问题，朴素贝叶斯算法对条件概率分布作出了独立性的假设**，通俗地讲就是说假设各个维度的特征$x_{1},x_{2},...,x_{n}$互相独立，在这个假设的前提上，条件概率可以转化为：

$P(x|y_{k})=P(x_{1},x_{2},...,x_{n}|y_{k})=\prod_{i=1}^{n}P(x_{i}|y_{k})  $  **【公式2】**

这样，参数规模就降到$\sum_{i=1}^{n}S_{i}k$

以上就是针对条件概率所作出的特征条件独立性假设，至此，先验概率$P(y_{k})$和条件概率$P(x|y_{k})$的求解问题就都解决了，那么我们是不是可以求解我们所要的后验概率$P(y_{k}|x)$了？

>这里思考两分钟

答案是肯定的。我们继续上面关于$P(y_{k}|x)$的推导，将【公式2】代入【公式1】得到：

$P(y_{k}|x)=\frac{P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}{\sum_{k}P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}$

于是朴素贝叶斯分类器可表示为：

$f(x)=argmax_{y_{k}} P(y_{k}|x)=argmax_{y_{k}} \frac{P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}{\sum_{k}P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})}$

因为对所有的$y_{k}$，上式中的分母的值都是一样的（为什么？注意到全加符号就容易理解了），所以可以忽略分母部分，朴素贝叶斯分类器最终表示为：

$f(x)=argmax P(y_{k})\prod_{i=1}^{n}P(x_{i}|y_{k})$

关于$P(y_{k})$，$P(x_{i}|y_{k})$的求解，有以下三种常见的模型.


#2. 三种常见的模型及编程实现

##2.1 多项式模型

当特征是离散的时候，使用多项式模型。多项式模型在计算先验概率$P(y_{k})$和条件概率$P(x_{i}|y_{k})$时，会做一些**平滑处理**，具体公式为：

$P(y_{k})=\frac{N_{y_{k}}+\alpha}{N+k\alpha}$

>N是总的样本个数，k是总的类别个数，$N_{y_{k}}$是类别为$y_{k}$的样本个数，$\alpha$是平滑值。

$P(x_{i}|y_{k})=\frac{N_{y_{k},x_{i}}+\alpha}{N_{y_{k}}+n\alpha}$

>$N_{y_{k}}$是类别为$y_{k}$的样本个数，n是特征的维数，$N_{y_{k},x_{i}}$是类别为$y_{k}$的样本中，第i维特征的值是$x_{i}$的样本个数，$\alpha$是平滑值。

当$\alpha=1$时，称作Laplace平滑，当$0<\alpha<1$时，称作Lidstone平滑，$\alpha=0$时不做平滑。

如果不做平滑，当某一维特征的值$x_{i}$没在训练样本中出现过时，会导致$P(x_{i}|y_{k})=0$，从而导致后验概率为0。加上平滑就可以克服这个问题。

### 2.1.1 举例
有如下训练数据，15个样本，2维特征$X^{1},X^{2}$，2种类别-1，1。给定测试样本$x=(2,S)^{T}$，判断其类别。

![这里写图片描述](http://img.blog.csdn.net/20150909084656149)

解答如下：

运用多项式模型，令$\alpha=1$

- 计算先验概率

![这里写图片描述](http://img.blog.csdn.net/20150909085100191)

- 计算各种条件概率

![这里写图片描述](http://img.blog.csdn.net/20150909085145105)

- 对于给定的$x=(2,S)^{T}$，计算：

![这里写图片描述](http://img.blog.csdn.net/20150909085219342)

由此可以判定y=-1。

###2.1.2 编程实现（基于Python，Numpy）

从上面的实例可以看到，当给定训练集时，我们无非就是先计算出所有的先验概率和条件概率，然后把它们存起来（当成一个查找表）。当来一个测试样本时，我们就计算它所有可能的后验概率，最大的那个对应的就是测试样本的类别，而后验概率的计算无非就是在查找表里查找需要的值。

我的代码就是根据这个思想来写的。定义一个MultinomialNB类，它有两个主要的方法：fit(X,y)和predict(X)。fit方法其实就是训练，调用fit方法时，做的工作就是构建查找表。predict方法就是预测，调用predict方法时，做的工作就是求解所有后验概率并找出最大的那个。此外，类的构造函数\__init__()中，允许设定$\alpha$的值，以及设定先验概率的值。具体代码及如下：

```
"""
Created on 2015/09/06

@author: wepon (http://2hwp.com)

API Reference: http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
"""
import numpy as np

class MultinomialNB(object):
        """
	    Naive Bayes classifier for multinomial models
        The multinomial Naive Bayes classifier is suitable for classification with
        discrete features
	
	    Parameters
        ----------
        alpha : float, optional (default=1.0)
                Setting alpha = 0 for no smoothing
		Setting 0 < alpha < 1 is called Lidstone smoothing
		Setting alpha = 1 is called Laplace smoothing 
        fit_prior : boolean
                Whether to learn class prior probabilities or not.
                If false, a uniform prior will be used.
        class_prior : array-like, size (n_classes,)
                Prior probabilities of the classes. If specified the priors are not
                adjusted according to the data.
		
	    Attributes
        ----------
        fit(X,y):
                X and y are array-like, represent features and labels.
                call fit() method to train Naive Bayes classifier.
        
        predict(X):
                
	
	"""
	
	def __init__(self,alpha=1.0,fit_prior=True,class_prior=None):
        self.alpha = alpha
		self.fit_prior = fit_prior
		self.class_prior = class_prior
		self.classes = None
		self.conditional_prob = None
		
	
	def _calculate_feature_prob(self,feature):
        values = np.unique(feature)
		total_num = float(len(feature))
		value_prob = {}
		for v in values:
			value_prob[v] = (( np.sum(np.equal(feature,v)) + self.alpha ) /( total_num + len(values)*self.alpha))
		return value_prob
		
		
	def fit(self,X,y):
		#TODO: check X,y
		
		self.classes = np.unique(y)
		#calculate class prior probabilities: P(y=ck)
		if self.class_prior == None:
            class_num = len(self.classes)
			if not self.fit_prior:
				self.class_prior = [1.0/class_num for _ in range(class_num)]  #uniform prior
			else:
				self.class_prior = []
				sample_num = float(len(y))
				for c in self.classes:
					c_num = np.sum(np.equal(y,c))
					self.class_prior.append((c_num+self.alpha)/(sample_num+class_num*self.alpha))
		
		#calculate Conditional Probability: P( xj | y=ck )
		self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
		for c in self.classes:
			self.conditional_prob[c] = {}
			for i in range(len(X[0])):  #for each feature
                feature = X[np.equal(y,c)][:,i]
				self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
		return self


    #given values_prob {value0:0.2,value1:0.1,value3:0.3,.. } and target_value
	#return the probability of target_value
	def _get_xj_prob(self,values_prob,target_value):
        return values_prob[target_value]
	
    #predict a single sample based on (class_prior,conditional_prob)
	def _predict_single_sample(self,x):
        label = -1
        max_posterior_prob = 0
                
		#for each category, calculate its posterior probability: class_prior * conditional_prob
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
			current_conditional_prob = 1.0
			feature_prob = self.conditional_prob[self.classes[c_index]]
			j = 0
			for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i],x[j])
				j += 1
			
			#compare posterior probability and update max_posterior_prob, label
			if current_class_prior * current_conditional_prob > max_posterior_prob:
				max_posterior_prob = current_class_prior * current_conditional_prob
				label = self.classes[c_index]
		return label
	
	#predict samples (also single sample)			
	def predict(self,X):
        #TODO1:check and raise NoFitError 
        #ToDO2:check X
        if X.ndim == 1:
                return self._predict_single_sample(X)
        else:
                #classify each sample 	
                labels = []
                for i in range(X.shape[0]):
                        label = self._predict_single_sample(X[i])
                        labels.append(label)
                return labels

```

我们用上面举的例子来检验一下，注意S,M,L我这里用4，5，6替换：

```
import numpy as np
X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
             ])
X = X.T
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

nb = MultinomialNB(alpha=1.0,fit_prior=True)
nb.fit(X,y)
print nb.predict(np.array([2,4]))#输出-1

```


##2.2 高斯模型
当特征是连续变量的时候，运用多项式模型就会导致很多$P(x_{i}|y_{k})=0$（不做平滑的情况下），此时即使做平滑，所得到的条件概率也难以描述真实情况。所以处理连续的特征变量，应该采用高斯模型。

###2.2.1  通过一个例子来说明：
[性别分类的例子](http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html)
[来自维基](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification)

下面是一组人类身体特征的统计资料。
| 性别 | 身高（英尺） | 体重（磅） |脚掌（英寸）|
| :-------------: |:-------------:| :-----:|:-----:|
|　男 　|　　6 　　|　　　　180　　|　　　12 |
|男 　|　　5.92　　|　　　190　　　|　　11   |
|男 　　|　5.58　　|　　　170　　　|　　12 |
|　　男 　　|　5.92　|　　　　165　　|　　　10 |
|　　女 　　|　5 　|　　　　　100　　　|　　6 |
|　　女 　|　　5.5 　　|　　　150　　|　　　8 |
|　　女 　|　　5.42　　|　　　130　|　　　　7 |
|　　女 　|　　5.75　　|　　　150　|　　　　9|

已知某人身高6英尺、体重130磅，脚掌8英寸，请问该人是男是女？
根据朴素贝叶斯分类器，计算下面这个式子的值。

    P(身高|性别) x P(体重|性别) x P(脚掌|性别) x P(性别)
   
这里的困难在于，由于身高、体重、脚掌都是连续变量，不能采用离散变量的方法计算概率。而且由于样本太少，所以也无法分成区间计算。怎么办？
这时，可以假设男性和女性的身高、体重、脚掌都是正态分布，通过样本计算出均值和方差，也就是得到正态分布的密度函数。有了密度函数，就可以把值代入，算出某一点的密度函数的值。
比如，男性的身高是均值5.855、方差0.035的正态分布。所以，男性的身高为6英尺的概率的相对值等于1.5789（大于1并没有关系，因为这里是密度函数的值，只用来反映各个值的相对可能性）。

![这里写图片描述](http://img.blog.csdn.net/20150909092824838)

对于脚掌和体重同样可以计算其均值与方差。有了这些数据以后，就可以计算性别的分类了。
```
   P(身高=6|男) x P(体重=130|男) x P(脚掌=8|男) x P(男) 
　　　　= 6.1984 x e-9
　　P(身高=6|女) x P(体重=130|女) x P(脚掌=8|女) x P(女) 
　　　　= 5.3778 x e-4
```
可以看到，女性的概率比男性要高出将近10000倍，所以判断该人为女性。


- 总结

**高斯模型假设每一维特征都服从高斯分布（正态分布）：**

$P(x_{i}|y_{k})=\frac{1}{\sqrt{2\pi\sigma_{y_{k},i}^{2}}}e^{-\frac{(x_{i}-\mu_{y_{k},i})^{2}}{2  \sigma_{y_{k},i}^{2}}}$

$\mu_{y_{k},i}$表示类别为$y_{k}$的样本中，第i维特征的均值。
$\sigma_{y_{k},i}^{2}$表示类别为$y_{k}$的样本中，第i维特征的方差。


###2.2.2 编程实现

高斯模型与多项式模型唯一不同的地方就在于计算 $ P( x_{i} | y_{k}) $，高斯模型假设各维特征服从正态分布，需要计算的是各维特征的均值与方差。所以我们定义GaussianNB类，继承自MultinomialNB并且重载相应的方法即可。代码如下：

```
#GaussianNB differ from MultinomialNB in these two method:
# _calculate_feature_prob, _get_xj_prob
class GaussianNB(MultinomialNB):
        """
        GaussianNB inherit from MultinomialNB,so it has self.alpha
        and self.fit() use alpha to calculate class_prior
        However,GaussianNB should calculate class_prior without alpha.
        Anyway,it make no big different

        """
        #calculate mean(mu) and standard deviation(sigma) of the given feature
        def _calculate_feature_prob(self,feature):
                mu = np.mean(feature)
                sigma = np.std(feature)
                return (mu,sigma)
        
        #the probability density for the Gaussian distribution 
        def _prob_gaussian(self,mu,sigma,x):
                return ( 1.0/(sigma * np.sqrt(2 * np.pi)) *
                        np.exp( - (x - mu)**2 / (2 * sigma**2)) )
        
        #given mu and sigma , return Gaussian distribution probability for target_value
        def _get_xj_prob(self,mu_sigma,target_value):
                return self._prob_gaussian(mu_sigma[0],mu_sigma[1],target_value)


```



##2.3 伯努利模型

与多项式模型一样，伯努利模型适用于离散特征的情况，所不同的是，伯努利模型中每个特征的取值只能是1和0(以文本分类为例，某个单词在文档中出现过，则其特征值为1，否则为0).

伯努利模型中，条件概率$P(x_{i}|y_{k})$的计算方式是：

当特征值$x_{i}$为1时，$P(x_{i}|y_{k})=P(x_{i}=1|y_{k})$；

当特征值$x_{i}$为0时，$P(x_{i}|y_{k})=1-P(x_{i}=1|y_{k})$；

###2.3.1 编程实现

伯努利模型和多项式模型是一致的，BernoulliNB需要比MultinomialNB多定义一个二值化的方法，该方法会接受一个阈值并将输入的特征二值化（1，0）。当然也可以直接采用MultinomialNB，但需要预先将输入的特征二值化。写到这里不想写了，编程实现留给读者吧。


##3 参考文献
- 《统计学习方法》，李航
- 《机器学习》
- [维基百科Sex classification](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification)
- [朴素贝叶斯的三个常用模型：高斯、多项式、伯努利](http://www.letiantian.me/2014-10-12-three-models-of-naive-nayes/)
- [朴素贝叶斯分类器的应用](http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html)
- [数学之美番外篇：平凡而又神奇的贝叶斯方法](http://blog.csdn.net/pongba/article/details/2958094)
