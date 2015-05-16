#coding:utf-8

'''
Author:wepon
Code:https://github.com/wepe

File:cnn.py
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
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


nb_epoch = 5
batch_size = 100
nb_class = 10
#加载数据
data, label = load_data()

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, nb_class)

#14layer, one "add" represent one layer
def create_model():
	model = Sequential()
	model.add(Convolution2D(4, 1, 5, 5, border_mode='valid')) 
	model.add(Activation('relu'))

	model.add(Convolution2D(8,4, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(poolsize=(2, 2)))

	model.add(Convolution2D(16, 8, 3, 3, border_mode='valid')) 
	model.add(Activation('relu'))
	model.add(MaxPooling2D(poolsize=(2, 2)))

	model.add(Flatten())
	model.add(Dense(16*4*4, 128, init='normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(128, nb_class, init='normal'))
	model.add(Activation('softmax'))
	return model


#############
#开始训练模型
##############
model = create_model()
sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
(X_train,X_val) = (data[0:30000],data[30000:])
(Y_train,Y_val) = (label[0:30000],label[30000:])
best_accuracy = 0.0
for e in range(nb_epoch):
    #shuffle the data each epoch
    num = len(Y_train)
    index = [i for i in range(num)]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    print('Epoch', e)
    print("Training...")
    batch_num = len(Y_train)/batch_size
    progbar = generic_utils.Progbar(X_train.shape[0])
    for i in range(batch_num):
        loss,accuracy = model.train(X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size],accuracy=True)
        progbar.add(batch_size, values=[("train loss", loss),("train accuracy:", accuracy)] )

    #save the model of best val-accuracy
    print("Validation...")
    val_loss,val_accuracy = model.evaluate(X_val, Y_val, batch_size=1,show_accuracy=True)
    if best_accuracy<val_accuracy:
        best_accuracy = val_accuracy
	cPickle.dump(model,open("./model.pkl","wb"))
