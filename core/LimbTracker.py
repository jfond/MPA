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
	load_video_filename = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\Set 8\\Removed_Gradient.avi"
	#load_video_filename = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\21072014\\Trial5\\PS3_Vid83.avi"


	cap = cv2.VideoCapture(load_video_filename)

	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 1,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 5)

	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
					  maxLevel = 3,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(100,3))
	
	scale_power = 1
	
	
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	old_gray = np.power(old_gray,scale_power)
	old_gray = Scale_Image(old_gray)

	frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	#Select mask in which we should look for the initial  good points to track - a.k.a. select limb to track
	#old_frame = frame.astype(np.uint8)
	#old_frame, timestamp = self.camera.read()
	pl.figure()
	pl.title("Select mask")
	pl.imshow(old_gray, cmap=mpl_cm.Greys_r)
	pts = []
	while not len(pts):
		pts = pl.ginput(0)
	pl.close()
	path = mpl_path.Path(pts)
	mask = np.zeros(np.shape(old_gray), dtype=np.uint8)
	ang_initial = np.zeros(np.shape(old_frame), dtype=np.float32)
	sum = np.float32(0)
	for ridx,row in enumerate(mask):
		for cidx,pt in enumerate(row):
			if path.contains_point([cidx, ridx]):
				mask[ridx,cidx] = 1
				
	# Apply mask to first frame and find good features in it
	old_gray *= mask
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	for n in range(len(p0)):
		if mask[p0[n][0][1],p0[n][0][0]] == 0:
			p0 = np.delete(p0,n,axis=0)


	# Create a mask image for drawing purposes
	line_mask = np.zeros_like(old_frame)

	points = np.zeros((frames,len(p0[0]),2))

	for n in range(frames):
		ret,frame = cap.read()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_gray = np.power(frame_gray,scale_power)
		frame_gray = Scale_Image(frame_gray)

		#print p0
		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		# Select good points
		
		good_new = p1[st==1]
		good_old = p0[st==1]
						
		if len(good_new) == 0:
			break
			
		points[n][0:len(good_new)] = good_new
			
		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			cv2.line(line_mask, (a,b),(c,d), color[i].tolist(), 1)
			cv2.circle(frame,(a,b),2,color[i].tolist(),-1)
			
		img = cv2.add(frame,line_mask)

		#time.sleep(5)
		cv2.imshow('frame',img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

		
			
		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

	print points
	angles = np.zeros(len(points))
	for n in range(n-1):
		angles[n] = np.arctan2(points[n+1][1]-points[n][1],points[n+1][0]-points[n][1])
		
	print angles
		
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
		