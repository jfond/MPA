import cv2
import cv2.cv as cv
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.pyplot as plt
import time
import os

def Main(save_file_location,video_filename,mask=None,Display_Images = True):

	cap = cv2.VideoCapture(video_filename)
	ret, frame = cap.read()
	prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	
	frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))	

	CIRCLE_CENTER_X = 160
	CIRCLE_CENTER_Y = 410
	CIRCLE_RADIUS = 220
	sum = 0
	ang_initial = np.zeros(np.shape(prvs), dtype=np.float32)
	if mask is None:
		mask,sum,ang_initial = Set_Mask(video_filename)
	else:
		for ridx,row in enumerate(mask):
			for cidx,pt in enumerate(row):
				if mask[ridx,cidx] != 0:
					sum +=1
					ang_initial[ridx,cidx] = np.arctan2(cidx-CIRCLE_CENTER_X,CIRCLE_CENTER_Y-ridx)
	



	ang_final = np.zeros(np.shape(prvs), dtype=np.float32)
	phase = []

	for n in range(frames-1):
		print "Calculating Wheel Angular Velocity: %i of %i frames complete"%(n+1,frames)
		ret, frame2 = cap.read()
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		next *= mask

		flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5,1,3,15,3,5,1)

		for ridx,row in enumerate(next):
			for cidx,pt in enumerate(row):
				if mask[ridx,cidx] == 1:
					ang_final[ridx,cidx] = np.arctan2(cidx+flow[ridx,cidx,0]-CIRCLE_CENTER_X,CIRCLE_CENTER_Y-ridx-flow[ridx,cidx,1])
					
		d_ang = ang_final - ang_initial
		wval = np.sum(d_ang)/sum
		phase.append(np.float32(-1*wval)) #Negative so that forward motion is positive
		
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('opticalfb.png',frame2)
			cv2.imwrite('opticalhsv.png',rgb)
		prvs = next

	phase[0] = 0
	phase = np.array(phase)
	plot = cv2.GaussianBlur(phase,(11,0),4) #Smooth the average in the time dimension
			 
	plt.plot(range(1,len(phase)+1),plot)
	np.save(os.path.join(save_file_location,"Wheel_Anglular_Velocity.npy"), plot)
	plt.savefig(os.path.join(save_file_location,"Wheel_Optical_Flow.png"))
	if Display_Images:
		plt.show()

	plt.clf()	
	cap.release()
	cv2.destroyAllWindows()
	

def Set_Mask(video_filename):
	
	cap = cv2.VideoCapture(video_filename)

	CIRCLE_CENTER_X = 160
	CIRCLE_CENTER_Y = 410
	CIRCLE_RADIUS = 220

	ret, frame = cap.read()

	frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = frame.astype(np.uint8)
	#frame, timestamp = self.camera.read()
	pl.figure()
	pl.title("Select Wheel Where No Part of the Mouse Touches Ever")
	pl.imshow(frame, cmap=mpl_cm.Greys_r)
	pts = []
	while not len(pts):
		pts = pl.ginput(0)
	pl.close()
	path = mpl_path.Path(pts)
	mask = np.zeros(np.shape(frame), dtype=np.uint8)
	ang_initial = np.zeros(np.shape(frame), dtype=np.float32)
	sum = np.float32(0)
	for ridx,row in enumerate(mask):
		for cidx,pt in enumerate(row):
			if path.contains_point([cidx, ridx]):
				mask[ridx,cidx] = 1
				sum +=1
				ang_initial[ridx,cidx] = np.arctan2(cidx-CIRCLE_CENTER_X,CIRCLE_CENTER_Y-ridx)
				
	return mask,sum,ang_initial
	
if __name__ == "__main__":
 	save_file_location = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\Set 7"
	video_filename = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\Set 7\\Gradient.avi"   
		
	Main(save_file_location,video_filename)		