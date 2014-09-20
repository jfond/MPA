import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.pyplot as plt
import time
import os
from scipy.signal import medfilt

def Main(load_video_filename,save_file_location,Automatically_Calculate_Projection_Angle=False,masks=None,Display_Images = True):

	FOURCC = cv2.cv.CV_FOURCC(*'XVID')	
	
	cap = cv2.VideoCapture(load_video_filename)

	save_file = os.path.join(save_file_location,"Motor Primitives.png")
	vector_savename = os.path.join(save_file_location,"Motor_Primitives_Data.npy")
	
	pt1 = (160,120)
	arrow_color = (255,255,255)
	frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	ret, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

	color_names_RBG = ['b','g','r','c','m','y','k']  #BGR
	
	if masks is not None:
		num_masks = np.shape(masks)[0]
	else:
		num_masks = 1
		masks = np.zeros((1,np.shape(prvs)[0],np.shape(prvs)[1]),dtype=np.uint8)
		masks[0] = (prvs!=0).astype(np.uint8)
	
	angle_list = np.zeros((num_masks,frames))
	mag_list = np.zeros((num_masks,frames))
	
	for n in range(frames-1):
		ret, frame2 = cap.read()
		next = cv2.cvtColor(frame2[:],cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)	
		for m in range(num_masks):
			
			idx = np.where(masks[m] != 0)
						
			sum_x = np.sum(flow[idx[0],idx[1],0])
			sum_y = np.sum(flow[idx[0],idx[1],1])
			
			
			mag = np.sqrt(np.power(sum_x,2)+np.power(sum_y,2))
			angle  = np.arctan2(sum_y,sum_x)
				
			mag/=200 #Just a value we guessed, it works well
			
			if Display_Images:
				src = cv.fromarray(next)
				pt2 = (160+int(mag*np.cos(angle)),120+int(mag*np.sin(angle)))
				cv.Line(src, pt1, pt2, arrow_color, thickness=1, lineType=8, shift=0) #And here open cv decided to use (x,y) rather than (y,x)....
				cv2.imshow('frame2',next)
			
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
			elif k == ord('s'):
				cv2.imwrite('opticalfb.png',frame2)
				cv2.imwrite('opticalhsv.png',rgb)
				
			#angle_list[0] = (angle_list[-1] + angle)%360
			mag_list[m,n] = mag
			angle_list[m,n] = (angle*180/np.pi)
				
			prvs = next
	
	median_angle = np.zeros((num_masks,frames))
	
	for m in range(num_masks):
		median_angle[m,:] = medfilt(medfilt(angle_list[m,:],11))

		if Automatically_Calculate_Projection_Angle:
			angle_offset = np.mean(median_angle[m,:])
		else:
			plt.hist(median_angle[m,:],bins=20)
			if Display_Images:
				plt.show()

			while True:
				input = raw_input("Angle Offset:")
				input = input.rstrip()

				try:
					input = int(input)
				except:
					pass
				
				if type(input) == int:
					break
				else:
					print "Please enter a valid number"
				
			angle_offset = input
			plt.clf()
		
		median_angle[m,:] -= angle_offset
		
	for m in range(num_masks):
		myplot = plt.plot(np.sign(median_angle[m,:])*mag_list[m,:],color_names_RBG[m%7])

	save_vector = np.zeros((np.shape(angle_list)[0],np.shape(angle_list)[1],2))
	save_vector[:,:,0] = angle_list
	save_vector[:,:,1] = mag_list
	np.save(vector_savename,save_vector)   
	

	plt.savefig(save_file)
	if Display_Images:
		plt.show()
			
	plt.clf()
	cap.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	load_video_filename = "C:\Users\Camera\Desktop\Video_Editing_Tools\Analysis\MOUSE 2014-01, NUMBER 64\\FINAL.avi"
	save_file_location = "C:\Users\Camera\Desktop\Video_Editing_Tools\Analysis\MOUSE 2014-01, NUMBER 64"

	Main(load_video_filename,save_file_location)		