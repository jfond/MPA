import numpy as np
import cv2
import cv2.cv as cv
import time

#cap = cv2.VideoCapture('C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\21072014\\Trial5\\PS3_Vid83.avi')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
A = int(cap.get(3))
B = int(cap.get(4))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

s,image = cap.read()
image = cv2.imread('C:\\Users\\Camera\\Desktop\\Median6.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

STD_img = np.zeros(np.shape(image))

#idx = np.where(image<128)

#image[idx[0],idx[1]] = 0
image = np.float64(image)
for x in np.linspace(-2,1.9,40):
	timage = np.power(image,x)
	maximum = np.amax(timage)
	timage = np.int64(timage/maximum * 255)
	print x
	
	cv2.imwrite('C:\Users\Camera\Desktop\Threshold Filtert.png', timage)
	time.sleep(2)