import cv2
import cv2.cv as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as pytime
import matplotlib.axes as axes
from pylab import plot
import os

cap = cv2.VideoCapture("C:\Users\Camera\Desktop\PS3_Vid1.avi")
height = int(cap.get(4))
width = int(cap.get(3))
frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
data = np.zeros(frames)

for n in range(frames):
	y, frame = cap.read()
	frame = frame[height-101:height,:int(width*2/3)]
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
	
	mean = np.mean(frame)
	data[n-1] = mean
	
fig = plt.figure()
ax = fig.add_subplot(121)

ax.scatter(range(frames),data,color='blue')

ax.set_ylabel('Mean Intensity')
ax.grid(True)

plt.show()
fig = None
ax = None
plt.clf()
print "Done"