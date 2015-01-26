"""
@author:wepon
@blog:http://blog.csdn.net/u012162613/article/details/43157801

"""

# -*- coding: utf-8 -*-
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

"""
定义Softmax回归模型
在deeplearning tutorial中，直接将LogisticRegression视为Softmax，
而我们所认识的二类别的逻辑回归就是当n_out=2时的LogisticRegression
"""
#参数说明：
#input，输入的一个batch，假设一个batch有n个样本(n_example)，则input大小就是(n_example,n_in)
#n_in,每一个样本的大小，MNIST每个样本是一张28*28的图片，故n_in=784
#n_out,输出的类别数，MNIST有0～9共10个类别，n_out=10 
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):

#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。WX+b  
#W和b都定义为theano.shared类型，这个是为了程序能在GPU上跑。
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，
#再作为T.nnet.softmax的输入，得到p_y_given_x
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率    
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，
#然后(n_example,n_out)矩阵的每一行都加b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

#params，模型的参数     
        self.params = [self.W, self.b]

#代价函数NLL
#因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),
#y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，
#则LP[T.arange(y.shape[0]),y]得到[LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]
#最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost
    def negative_log_likelihood(self, y):  
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

#batch的误差率
    def errors(self, y):
        # 首先检查y与y_pred的维度是否一样，即是否含有相等的样本数
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率
        #举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]
        #则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等
        #故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

"""
加载MNIST数据集
"""
def load_data(dataset):
    # dataset是数据集的路径，程序首先检测该路径下有没有MNIST数据集，没有的话就下载MNIST数据集
    #这一部分就不解释了，与softmax回归算法无关。
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'
#以上是检测并下载数据集mnist.pkl.gz，不是本文重点。下面才是load_data的开始
    
#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
#主要用到python里的gzip.open()函数,以及 cPickle.load()。
#‘rb’表示以二进制可读的方式打开文件
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
   

#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

"""
将该模型应用于MNIST
"""
def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
#加载数据
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
#计算有多少个minibatch，因为我们的优化算法是MSGD，是一个batch一个batch来计算cost的
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # 开始建模            #
    ######################
    print '... building the model'


#设置变量，index表示minibatch的下标，x表示训练样本，y是对应的label
    index = T.lscalar()  
    x = T.matrix('x') 
    y = T.ivector('y') 
    
    
#定义分类器，用x作为input初始化。
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)


#定义代价函数，用y来初始化，而其实还有一个隐含的参数x在classifier中。
#这样理解才是合理的，因为cost必须由x和y得来，单单y是得不到cost的。
    cost = classifier.negative_log_likelihood(y)


#这里必须说明一下theano的function函数，givens是字典，其中的x、y是key，冒号后面是它们的value。
#在function被调用时，x、y将被具体地替换为它们的value，而value里的参数index就是inputs=[index]这里给出。
#下面举个例子：
#比如test_model(1)，首先根据index=1具体化x为test_set_x[1 * batch_size: (1 + 1) * batch_size]，
#具体化y为test_set_y[1 * batch_size: (1 + 1) * batch_size]。然后函数计算outputs=classifier.errors(y)，
#这里面有参数y和隐含的x，所以就将givens里面具体化的x、y传递进去。
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }

# 计算各个参数的梯度
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

#更新的规则，根据梯度下降法的更新公式
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

#train_model跟上面分析的test_model类似，只是这里面多了updatas，更新规则用上面定义的updates 列表。   
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # 开始训练     #
    ###############
    print '... training the model'
   
    patience = 5000  
    patience_increase = 2 
#提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss   
    improvement_threshold = 0.995  
#这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。
   validation_frequency = min(n_train_batches, patience / 2)
                                

    best_validation_loss = numpy.inf   #最好的验证集上的loss，最好即最小。初始化为无穷大
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    
#下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。
#for循环是遍历一个个batch，一次一个batch地训练。for循环体里会用train_model(minibatch_index)去训练模型，
#train_model里面的updatas会更新各个参数。
#for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，
#如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，
#则更新best_validation_loss和best_iter，同时在testset上测试。
#如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。
#当达到最大步数n_epoch时，或者patience<iter时，结束训练
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

#while循环结束
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
