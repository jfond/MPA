import cv2
import cv2.cv as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as pytime
import matplotlib.axes as axes
from pylab import plot
import os

#Try to sync videos according to the luminosity changes

Number_of_videos = 32
test_vector_length = 100 #must be even number

Data = np.empty((Number_of_videos,2,759))
Data[:] = None
Offset = np.zeros(Number_of_videos)

Show_Images = True

for n in range(1,Number_of_videos+1):
	
	print n
	
	cap1 = cv2.VideoCapture("C:\Users\Camera\Desktop\GtHUb\Two-Cameras\Data\AG051514-01\Synchronization\Trial 1\PG_Vid%04d.avi"%n)
	#cap2 = cv2.VideoCapture("C:\Users\Camera\Desktop\GtHUb\Two-Cameras\Data\AG051514-01\Synchronization\Trial 1\PS3_Vid%i.avi"%n)
	cap2 = cv2.VideoCapture("C:\Users\Camera\Desktop\PS3_Vid1.avi")
	
	height1 = int(cap1.get(4))
	height2 = int(cap2.get(4))
	
	width1 = int(cap1.get(3))
	width2 = int(cap2.get(3))
	
	frames_total = min((int(cap1.get(cv.CV_CAP_PROP_FRAME_COUNT)),int(cap2.get(cv.CV_CAP_PROP_FRAME_COUNT)))) - 5
	
	print str(frames_total)
	
	data_points_temp1 = np.zeros(frames_total)
	data_points_temp2 = np.zeros(frames_total)
		
	data_points1 = np.zeros(frames_total-11)	
	data_points2 = np.zeros(frames_total-11)	
		
	for m in range(frames_total):
	
		#if m%10 == 0:
			#print m
		
		y, frame1 = cap1.read()
		y, frame2 = cap2.read()
	
		img1 = frame1[height1-49:height1,(width1/2)-25:(width1/2)+25]
		img2 = frame2[height2-49:height2,(width2/2)-25:(width2/2)+25]
		
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)	
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)	
		
		if m > 10:
			
			wval1 = np.mean(img1)
			wval2 = np.mean(img2)
			
			data_points1[m-10] = wval1
			data_points2[m-10] = wval2
			
			#data_points_temp1[m-10] = int(wval1>50)
			#data_points_temp2[m-10] = int(wval2>50) 
			
			#for l in range(0,len(data_points_temp1)-11):
			#	data_points1[l] = np.mean(data_points_temp1[l:l+10])
			#	data_points2[l] = np.mean(data_points_temp2[l:l+10])
			
	cap1.release()
	cap2.release()			
	
	#Data[n-1][0][:] = data_points1
	#Data[n-1][1][:] = data_points2

	
	if Show_Images:	
		fig = plt.figure()
		ax = fig.add_subplot(121)
		
		ax.scatter(range(len(data_points1)),data_points1,color='blue')
		
		ax.set_ylabel('Point Grey Video')
		ax.grid(True)

		plt.show()
		fig = None
		ax = None
		plt.clf()
		pytime.sleep(0.1)
	
	
	if Show_Images:	
		fig = plt.figure()
		ax = fig.add_subplot(121)
		
		ax.scatter(range(len(data_points2)),data_points2,color='green')
		
		ax.set_ylabel('Playstation 3 Video')
		ax.grid(True)

		plt.show()
		fig = None
		ax = None
		plt.clf()
		pytime.sleep(0.1)

	if Show_Images:		
		fig = plt.figure()
		ax = fig.add_subplot(121)
		
		ax.scatter(data_points1, data_points2, color = 'red')
		
		ax.set_ylabel('Playstation 3 Video')
		ax.set_xlabel('Point Grey Video')
	
		plt.show()
		fig = None
		ax = None
		plt.clf()
		pytime.sleep(0.1)
	
	
	bot = int(test_vector_length/2)
	top = int(test_vector_length/2)
	
	length = len(data_points1)
	test_vector = data_points1[int(length/2)-bot:int(length/2)+top]
	test_vector = test_vector / (np.sqrt(np.dot(test_vector,test_vector)))
	
	magnitude = np.zeros(length-test_vector_length)
	
	for counter in range(0,length-test_vector_length):
		sliding_vector = data_points2[counter:counter+test_vector_length]
		sliding_vector = sliding_vector / (np.sqrt(np.dot(sliding_vector,sliding_vector)))
		
		magnitude[counter] = np.dot(sliding_vector, test_vector)
	
	if Show_Images:
		fig = plt.figure()
		ax = fig.add_subplot(121)
	
		xshift = int((length-test_vector_length)/2)
	
		ax.scatter(range(-xshift,-xshift + len(magnitude)), magnitude, color = 'purple')
	
		ax.set_xlabel('Offset')
		ax.set_ylabel('Similarity')	
	
		plt.show()
		fig = None
		ax = None
		plt.clf()
		pytime.sleep(0.1)
	
	Offset[n-1] = np.where(magnitude==max(magnitude))[0][0]
	
	
for offset in Offset:
	print offset
	
fig = plt.figure()
ax = fig.add_subplot(121)

ax.scatter(range(len(Offset)), Offset, color = 'black')

ax.set_xlabel('Video Number')
ax.set_ylabel('Offset Value')	

print str(np.mean(Offset))
print str(np.std(Offset))

plt.show()
fig = None
ax = None
plt.clf()
pytime.sleep(0.1)