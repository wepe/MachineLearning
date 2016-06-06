#coding:utf-8

'''
Author:wepon
Code:https://github.com/wepe

File:cnn.py
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py


2016.06.06更新：
这份代码是keras开发初期写的，当时keras还没有现在这么流行，文档也还没那么丰富，所以我当时写了一些简单的教程。
现在keras的API也发生了一些的变化，建议及推荐直接上keras.io看更加详细的教程。

'''
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_data
import random,cPickle
from keras.callbacks import EarlyStopping
import numpy as np

np.random.seed(1024)  # for reproducibility


#加载数据
data, label = load_data()


#label为0~9共10个类别，keras要求形式为binary class matrices,转化一下，直接调用keras提供的这个函数
nb_class = 10
label = np_utils.to_categorical(label, nb_class)


def create_model():
	model = Sequential()
	model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=(1,28,28))) 
	model.add(Activation('relu'))

	model.add(Convolution2D(8,3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(16,3, 3, border_mode='valid')) 
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, init='normal'))
	model.add(Activation('relu'))

	model.add(Dense(nb_class, init='normal'))
	model.add(Activation('softmax'))
	return model


#############
#开始训练模型
##############
model = create_model()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
(X_train,X_val) = (data[0:30000],data[30000:])
(Y_train,Y_val) = (label[0:30000],label[30000:])

#使用early stopping返回最佳epoch对应的model
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, batch_size=100,validation_data=(X_val, Y_val),nb_epoch=5,callbacks=[early_stopping])
cPickle.dump(model,open("./model.pkl","wb"))
