import cv2
import cv2.cv as cv
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.pyplot as plt
import time
import os

def Main():
	save_file_root = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\Set 8\\Limb_Movement"
	load_video_filename = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\Set 8\\Median.avi"
	#load_video_filename = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\21072014\\Trial5\\PS3_Vid83.avi"

	cap = cv2.VideoCapture(load_video_filename)
	
	ret, prvs = cap.read()
	prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

	frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	#Select mask in which we should look for the initial  good points to track - a.k.a. select limb to track
	pl.figure()
	pl.title("Select mask")
	pl.imshow(prvs, cmap=mpl_cm.Greys_r)
	pts = []
	while not len(pts):
		pts = pl.ginput(0)
	pl.close()
	path = mpl_path.Path(pts)
	mask = np.zeros(np.shape(prvs), dtype=np.uint8)
	for ridx,row in enumerate(mask):
		for cidx,pt in enumerate(row):
			if path.contains_point([cidx, ridx]):
				mask[ridx,cidx] = 1
	
	for n in range(frames):
		ret,next = cap.read()
		next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5,3,10,3,5,1.2,0)

		distt = 3
		threshold = 100
		new_mask = np.zeros(np.shape(prvs), dtype=np.uint8)
		for ridx,row in enumerate(mask):
			for cidx,pt in enumerate(row):
				if (mask[ridx,cidx]==1):	
					y = ridx+int(round(flow[ridx,cidx,1]))
					#print int(round(flow[ridx,cidx,1]))
					x = cidx+int(round(flow[ridx,cidx,0]))
					if int(round(flow[ridx,cidx,1])) < 10 and int(round(flow[ridx,cidx,0])) < 10:
						tmask = next[y-distt:y+distt + 1,x-distt:x+distt+1]
						#print tmask
						idx = np.where(tmask > threshold)
						#print idx[0]+y-distt
						new_mask[idx[0] + y - distt,idx[1] + x - distt] = 1
			
		mask = new_mask
		next*=new_mask
		
		cv2.imwrite("C:\\Users\\Camera\\Desktop\\TEST.png",next)
		cv2.imshow('Limb',next)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	prvs = next
	cv2.destroyAllWindows()
	cap.release()


def Scale_Image(image): #Scales the input image to the range:(0,255)
	min = np.amin(image)
	max = np.amax(image)
	
	image -= min
	image = np.float16(image) * (np.float16(255) / np.float16(max-min))
	image = np.uint8(image)

	return image

	
if __name__ == "__main__":
    Main()				
		