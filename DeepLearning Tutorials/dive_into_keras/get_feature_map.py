"""
Author:wepon
Code:https://github.com/wepe

File: get_feature_map.py
	1.  visualize feature map of Convolution Layer, Fully Connected layer
	2.  rewrite the code so you can treat CNN as feature extractor, see file: cnn-svm.py
"""
from __future__ import print_function
import cPickle,theano
from data import load_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#load the saved model
model = cPickle.load(open("model.pkl","rb"))

#define theano funtion to get output of  FC layer
get_feature = theano.function([model.layers[0].input],model.layers[11].get_output(train=False),allow_input_downcast=False) 

#define theano funtion to get output of  first Conv layer 
get_featuremap = theano.function([model.layers[0].input],model.layers[2].get_output(train=False),allow_input_downcast=False) 


data, label = load_data()

# visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images
feature = get_feature(data[0:10])  #visualize these images's FC-layer feature
plt.imshow(feature,cmap = cm.Greys_r)
plt.show()

#visualize feature map of Convolution Layer
num_fmap = 4	#number of feature map
for i in range(num_fmap):
	featuremap = get_featuremap(data[0:10])
	plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
	plt.show()
