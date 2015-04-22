#coding:utf-8
"""
讲文件下的所有图像写进csv文件，一行代表一张图
"""
import os
from PIL import Image
import numpy as np
import csv

#获取图像标签，以jaffe数据库为例。
def getlabel(img_name):
	face_expression = img_name.split('.')[1]   
	face_expression = face_expression[0:2]     
	table={'HA':1,'AN':2,'SU':3,'FE':4,'DI':5,'SA':6,'NE':7}
	return table.get(face_expression)


f = csv.writer(open("trainlbp.csv","wb"))


direction = "./jaffe"
img_list = os.listdir(direction)
for imgname in img_list:
	img = Image.open(direction+imgname)
	width,height = img.size
	data = np.empty((width*height+1))

	data[0] = getlabel(imgname)
	img_mat = np.array(img,dtype="float")
	img_mat = img_mat.flatten()
	data [1:] = img_mat
	#write into file
	f.writerow(data)








