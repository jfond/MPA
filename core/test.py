import cv2
import numpy as np
import cv2.cv as cv
import time
import matplotlib.pyplot as plt

framePerVid = 80
cap = cv2.VideoCapture('C:\\Users\\Camera\\Desktop\\Edge.avi')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
A = int(cap.get(3))
B = int(cap.get(4))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))

out = cv2.VideoWriter('C:\Users\Camera\Desktop\Compressed Edge.avi',fourcc, 10.0, (A,B),0)


for n in range(100):
	s,image = cap.read()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	image = image.astype(np.uint8)
	cv2.imshow('hi',image)
	
	out.write(image)
		
out.release()
cap.release()
