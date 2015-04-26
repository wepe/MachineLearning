#libsvm和liblinear的使用总结

0.安装方法
---

unix系统下的安装方法：到[官网](http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html)下载源包（目前最新版本为libsvm-3.20、liblinear-1.96），解压后，打开终端进入makefile所在的目录,键入make即可。

以下为一些基本的使用命令，ubuntu系统下。


1.生成符合要求的数据格式，以图像数据为例
---

1. 从图像库得到csv文件 （csv文件里每一行存储一张图：label,feat1,feat2,.....），在终端下键入：

		python gen_datafile.py 

	>注：gen_datafile.py是我自己写的一个python脚本，放在我的[github](https://github.com/wepe/MachineLearning/tree/master/SVM/libsvm%20liblinear-usage)

2. 编译convert.c生成可执行文件 a.out，在终端下键入：

		gcc convert.c

	>注：convert.c放在我的[github](https://github.com/wepe/MachineLearning/tree/master/SVM/libsvm%20liblinear-usage)

3. 用上面得到的csv文件和a.out文件生成libsvm格式的文件, 在终端下键入:

		./a.out csvfile > targetfile
	>注：targetfile是存放最终数据的文件。

2.训练模型的命令
---

 在终端下切换到目录liblinear-1.96或libsvm-3.20，然后键入以下命令，会提示具体用法：

	./svm-train (liblinear为./train)
	./svm-predict (liblinear为./predict)
	./svm-scale （数据缩放）
 


3.tools中easy.py的使用
----
easy.py是一条龙服务，从data scaling到参数选取都帮你做。

需要先安装gnuplot，安装命令：

	sudo apt-get install gnuplot-x11

之后键入：

	python easy.py training_file [testing_file]

4.tools中grid.py的使用:
---

grid.py用于自动搜索参数。用法，在终端下键入：

	pyhton grid.py [grid_options] [svm_options] dataset

>要查看options的具体信息，可以先不带参数地键入 pyhton grid.py，这是libsvm的通用方法。



5.tools中subset.py的使用
--
subset.py用于分割数据集。用法：

	Usage: subset.py [options] dataset subset_size [output1] [output2]

	This script randomly selects a subset of the dataset.

	options:
	-s method : method of selection (default 0)
	     0 -- stratified selection (classification only)
	     1 -- random selection


例如要随机选取dataset中的2000个样本作为trainset，剩下的作为testset，则键人：

	python subset.py  dataset 2000 trainset testset



6.tools中checkdata.py的使用
--
checkdata.py检查数据格式符不符合要求。键入：

	python checkdata.py dataset

7.其他：
---

- 使用交叉验证是不能生成model文件的？（我使用过程中发现不能，不知道是不是真的不能）



- 训练完的结果解读（选自网友博文）：

		optimization finished, #iter = 162

		nu = 0.431029

		obj = -100.877288, rho = 0.424462

		nSV = 132, nBSV = 107

		Total nSV = 132

		　　其中，#iter为迭代次数，nu 是你选择的核函数类型的参数，obj为SVM文件转换为的二次规划求解得到的最小值，rho为判决函数的偏置项b，nSV 为标准支持向量个数(0<a[i]<c)，nBSV为边界上的支持向量个数(a[i]=c)，Total nSV为支持向量总个数（对于两类来说，因为只有一个分类模型Total nSV = nSV，但是对于多类，这个是各个分类模型的nSV之和）。

		　　在目录下，还可以看到产生了一个train.model文件，可以用记事本打开，记录了训练后的结果。

			  svm_type c_svc                     //所选择的svm类型，默认为c_svc

			  kernel_type rbf                       //训练采用的核函数类型，此处为RBF核

			  gamma 0.0769231                   //RBF核的参数γ

			  nr_class 2                               //类别数，此处为两分类问题

			  total_sv 132                           //支持向量总个数

			  rho 0.424462                          //判决函数的偏置项b

			  label 1 -1                                 //原始文件中的类别标识

			  nr_sv 64 68                           //每个类的支持向量机的个数

			  SV                                          //以下为各个类的权系数及相应的支持向量

		   1 1:0.166667 2:1 3:-0.333333 … 10:-0.903226 11:-1 12:-1 13:1

		   0.5104832128985164 1:0.125 2:1 3:0.333333 … 10:-0.806452 12:-0.333333 13:0.5

		   ………..

		   -1 1:-0.375 2:1 3:-0.333333…. 10:-1 11:-1 12:-1 13:1

		    -1 1:0.166667 2:1 3:1 …. 10:-0.870968 12:-1 13:0.5


