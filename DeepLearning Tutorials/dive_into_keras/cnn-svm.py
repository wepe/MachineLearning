'''
Author:wepon
Code:https://github.com/wepe

File: cnn-svm.py
'''
from __future__ import print_function
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from data import load_data
import random


def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

if __name__ == "__main__":
    #load data
    data, label = load_data()
    #shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    
    (traindata,testdata) = (data[0:30000],data[30000:])
    (trainlabel,testlabel) = (label[0:30000],label[30000:])
    #use origin_model to predict testdata
    origin_model = cPickle.load(open("model.pkl","rb"))
    pred_testlabel = origin_model.predict_classes(testdata,batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(" Origin_model Accuracy:",accuracy)
    #define theano funtion to get output of FC layer
    get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    #train svm using FC-layer feature
    svc(feature[0:30000],label[0:30000],feature[30000:],label[30000:])
