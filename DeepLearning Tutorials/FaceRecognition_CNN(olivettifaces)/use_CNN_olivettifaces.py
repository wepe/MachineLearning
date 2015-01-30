# -*-coding:utf8-*-#
"""
本程序实现的功能：
在train_CNN_olivettifaces.py中我们训练好并保存了模型的参数，利用这些保存下来的参数来初始化CNN模型，
这样就得到一个可以使用的CNN系统，将人脸图输入这个CNN系统，预测人脸图的类别。

@author:wepon(http://2hwp.com)
讲解这份代码的文章：http://blog.csdn.net/u012162613/article/details/43277187
"""

import os
import sys
import cPickle

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


#读取之前保存的训练参数
#layer0_params~layer3_params都是包含W和b的,layer*_params[0]是W，layer*_params[1]是b
def load_params(params_file):
    f=open(params_file,'rb')
    layer0_params=cPickle.load(f)
    layer1_params=cPickle.load(f)
    layer2_params=cPickle.load(f)
    layer3_params=cPickle.load(f)
    f.close()
    return layer0_params,layer1_params,layer2_params,layer3_params

#读取图像，返回numpy.array类型的人脸数据以及对应的label
def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype='float64')/256

    faces=numpy.empty((400,2679))
    for row in range(20):
	   for column in range(20):
		faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

    label=numpy.empty(400)
    for i in range(40):
	label[i*10:i*10+10]=i
    label=label.astype(numpy.int)
    
    return faces,label



"""
train_CNN_olivettifaces中的LeNetConvPoolLayer、HiddenLayer、LogisticRegression是随机初始化的
下面将它们定义为可以用参数来初始化的版本
"""
class LogisticRegression(object):
    def __init__(self, input, params_W,params_b,n_in, n_out):
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

	
#卷积+采样层（conv+maxpooling）
class LeNetConvPoolLayer(object):
    def __init__(self,  input,params_W,params_b, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # 子采样
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


"""
用之前保存下来的参数初始化CNN，就得到了一个训练好的CNN模型，然后使用这个模型来测图像
注意：n_kerns跟之前训练的模型要保持一致。dataset是你要测试的图像的路径，params_file是之前训练时保存的参数文件的路径
"""
def use_CNN(dataset='olivettifaces.gif',params_file='params.pkl',nkerns=[5, 10]):   
    
    #读取测试的图像，这里读取整个olivettifaces.gif，即全部样本，得到faces、label
    faces,label=load_data(dataset)
    face_num = faces.shape[0]   #有多少张人脸图
  
    #读入参数
    layer0_params,layer1_params,layer2_params,layer3_params=load_params(params_file)
    
    x = T.matrix('x')  #用变量x表示输入的人脸数据，作为layer0的输入

    ######################
    #用读进来的参数初始化各层参数W、b
    ######################
    layer0_input = x.reshape((face_num, 1, 57, 47)) 
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        image_shape=(face_num, 1, 57, 47),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    layer1 = LeNetConvPoolLayer(
        input=layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(face_num, nkerns[0], 26, 21),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=nkerns[1] * 11 * 8,
        n_out=2000,      
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=2000, n_out=40)   
     
    #定义theano.function，让x作为输入，layer3.y_pred（即预测的类别）作为输出
    f = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer3.y_pred
    )
    
    #预测的类别pred
    pred = f(faces)
    

    #将预测的类别pred与真正类别label对比，输出错分的图像
    for i in range(face_num): 
	 if pred[i] != label[i]:
                print('picture: %i is person %i, but mis-predicted as person %i' %(i, label[i], pred[i]))


if __name__ == '__main__':
	use_CNN()



"""一点笔记,对theano.function的理解，不一定正确，后面深入理解了再回头看看

在theano里面，必须通过function定义输入x和输出，然后调用function，才会开始计算，比如在use_CNN里面，在定义layer0时，即使将faces作为输入，将layer1～layer3定义好后，也无法直接用layer3.y_pred来获得所属类别。
因为在theano中，layer0～layer3只是一种“图”关系，我们定义了layer0～layer3，也只是创建了这种图关系，但是如果没有funtion，它是不会计算的。

这也是为什么要定义x的原因：
    x = T.matrix('x')

然后将变量x作为layer0的输入。
最后，定义一个function：
f = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer3.y_pred
    )

将x作为输入，layer3.y_pred作为输出。
当调用f(faces)时，就获得了预测值

"""