import numpy as np
import cv2
import cv2.cv as cv
import sys
import time as pytime

# Define the codec and create VideoWriter object
fourcc = cv.CV_FOURCC(*'XVID')
cap = cv2.VideoCapture('C:\Users\Camera\Desktop\GtHUb\Two-Cameras\Data\Trial16\Synchronized_Vid1.avi')
width = int(cap.get(3))
height = int(cap.get(4))
frames_in_video = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter('C:\Users\Camera\Desktop\Synchronized_Vid1_0.1x_speed.avi',fourcc, 12, (width,height),0)

for m in range(frames_in_video):
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = frame.astype(np.uint8)
	
	out.write(frame)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
		break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()