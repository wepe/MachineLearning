#coding:utf-8
"""
Author:wepon
Code:https://github.com/wepe

File: data.py

download data here: http://pan.baidu.com/s/1qCdS6

"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，如果是将彩色图作为输入,则将1替换为3,图像大小28*28
def load_data():
	data = np.empty((42000,1,28,28),dtype="float32")
	label = np.empty((42000,),dtype="uint8")
	imgs = os.listdir("./mnist")
	num = len(imgs)
	for i in range(num):
		img = Image.open("./mnist/"+imgs[i])
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = arr
		label[i] = int(imgs[i].split('.')[0])
	#归一化和零均值化
	scale = np.max(data)
	data /= scale
	mean = np.std(data)
	data -= mean
	return data,label








