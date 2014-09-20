import cv2
import numpy as np
import cv2.cv as cv
import time
import matplotlib.pyplot as plt
import os
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.pyplot as plt

def Main():
	
	#Filename = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\18072014\Trial1\\PS3_Vid64.avi"
	Filename = "C:\Users\Camera\Desktop\Video_Editing_Tools\Analysis\MOUSE 2014-01, NUMBER 64\\MEDIAN_GRADIENT.avi"
	save_file = "C:\Users\Camera\Desktop\Video_Editing_Tools\Analysis\MOUSE 2014-01, NUMBER 64\\FINAL.avi"
	
	Analyze_Frame = True
	threshold = 100
	
	while True:
		#cv2.imshow("Image to be edited",image)
		print "What editing would you like to do?"
		print "  1: Median Filter"
		print "  2: Gaussian Smoothing"
		print "  3: Gradient Map"
		print "  4: Gradient Map with Edge Thinning"
		print "  5: Remove Part of Video"
		print "  6: Threshold"
		print "  7: Quit"
		input = raw_input("Your Choice:")
		input = input.rstrip()
		
		try:
			input = int(input)
		except:
			pass
			
		if type(input) == int and input > 0 and input <=7:
			if input == 1:
				size = querySize()
				image = Median_Filter(size,Filename,save_file)
			elif input == 2:
				size = querySize()
				image = Gaussian_Filter(size, save_file,Filename)
			elif input == 3:
				image = Gradient_Filter(save_file,Filename)
			elif input == 4:
				image = Canny(save_file,Filename)
			elif input == 5:
				image = ROI_Remove(save_file,Filename)
			elif input == 6:
				Threshold_IMG(save_file,Filename,threshold)
			elif input == 7:
				break
			else:
				print "Your response is invalid. Please enter a valid response"
		else:
			print "Please enter a valid number"
			
def querySize():
	while True:
		input = raw_input("Filter Radius:")
		input = input.rstrip()
	
		try:
			input = int(input)
		except:
			pass
		
		if type(input) == int:
			return input
		else:
			print "Please enter a valid number"
			
def Median_Filter(size,Filename,save_file):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	median_image = np.zeros((height,width))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 20.0, (width,height),0)
	for n in range(frames):
		s,image = Video.read()
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		median_image = cv2.medianBlur(image, size)
		
		median_image = Scale_Image(median_image)				
		out.write(median_image)
			
	return median_image
			
def Gaussian_Filter(size,save_file,Filename):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	smooth_image = np.zeros((height,width))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 10.0, (width,height),0)
	while True:
		input = raw_input("Smoothing strength:")
		input = input.rstrip()
	
		try:
			input = int(input)
		except:
			pass
		
		if type(input) == int:
			t=input
			break
		else:
			print "Please enter a valid number"
	
	for n in range(frames):
		s,image = Video.read()
		smooth_image = cv2.GaussianBlur(image,(size,size),t).astype(np.uint8)
		smooth_image = cv2.cvtColor(smooth_image, cv2.COLOR_BGR2GRAY)
		out.write(smooth_image)
	
	out.release()
	Video.release()
	return smooth_image
	
		
def Gradient_Filter(save_file,Filename):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	smooth_image = np.zeros((height,width))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 20.0, (width,height),0)
	grad_img = np.zeros((height,width))
	dx = [[-1,0,1],[-2,0,2],[-1,0,1]]
	dy = [[-1,-2,-1],[0,0,0],[1,2,1]]
	for n in range(frames):
		s,image = Video.read()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		Gx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)
		Gy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)
		
			
		# for ridx,row in enumerate(image):
			# for cidx, pt in enumerate(row):
				# if (ridx>= 1) and (ridx< height-1) and (cidx>= 1) and (cidx< width-1):

					# #point = np.float32(image[ridx-1:ridx+2,cidx-1:cidx+2])
					# #Gx = np.sum(point*dx)
					# #Gy = np.sum(point*dy)
					# Gx = np.power(Gx,2)
					# Gy = np.power(Gy,2)
					# gradient = np.sqrt(Gx+Gy)
					# grad_img[ridx,cidx] = gradient
					

		#grad_img = np.clip(grad_img)
		Gx = np.power(Gx,2)
		Gy = np.power(Gy,2)
		gradient = np.sqrt(Gx+Gy)
		abs_grad = np.abs(gradient)
		#print np.amax(abs_grad)
		abs_grad = np.float32(abs_grad) * 255.0 / 4000.0
		abs_grad = np.clip(abs_grad,0,255)
		uint8_grad = np.uint8(abs_grad)
		
		grad_img = uint8_grad	
		
		#grad_img  = cv2.Laplacian(image,cv2.cv2.CV_16S,ksize=5)
		grad_img = Scale_Image(grad_img)
		
		out.write(grad_img)
	return grad_img
	
	out.release()
	Video.release()

def Grid(File_directory, Filename, save_directory, grid_size):
	file = os.path.join(File_directory,Filename+".avi")
	savefile = os.path.join(save_directory,"Neuron_Grid.png")
	Video = cv2.VideoCapture(file)
	height = int(Video.get(4))
	width = int(Video.get(3))
	xlen = int(width/grid_size[1])
	ylen = int(height/grid_size[0])
	s,image = Video.read()
	
	for y in range(1,ylen+1):
		for x in range(xlen): #And here open cv decided to use (x,y) rather than (y,x) for cv2.Line
			cv2.putText(image, str(((y-1)*xlen)+x), ((((x*grid_size[1])+3),(y*grid_size[0])-7)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,0,255))	
			#The offset vaules (4,-7) were found by looking by eye and seeing what looked best
	image = cv.fromarray(image)			
			
	for y in range(1,ylen):
		cv.Line(image, (0,int(grid_size[1]*y)), (width,int(grid_size[1]*y)), (0,255,0), thickness=2, lineType=8, shift=0)			
	for x in range(1,xlen):
		cv.Line(image, (int(grid_size[1]*x),0), (int(grid_size[1]*x),height), (0,255,0), thickness=2, lineType=8, shift=0)			
		
	image = np.asarray( image[:,:] )
	
	cv2.imwrite(savefile,image)

	
def Scale_Image(image): #Scales the input image to the range:(0,255)
	min = np.amin(image)
	max = np.amax(image)
	
	image -= min
	image = np.float16(image) * (np.float16(255) / np.float16(max-min))
	image = np.uint8(image)

	return image
	
def Canny(save_file,Filename): #Canny edge detection implementing bilinear pixel interpolation
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	smooth_image = np.zeros((height,width))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 20.0, (width,height),0)
	Canny_image = np.zeros((height,width))
	high_threshold = 200
	low_threshold = 100
	while True:
		input = raw_input("High Threshold:")
		input = input.rstrip()
	
		try:
			input = int(input)
		except:
			pass
		
		if type(input) == int:
			high_threshold=input
			break
		else:
			print "Please enter a valid number"
			
	while True:
		input = raw_input("Low Threshold:")
		input = input.rstrip()
	
		try:
			input = int(input)
		except:
			pass
		
		if type(input) == int:
			low_threshold=input
			break
		else:
			print "Please enter a valid number"		

	for n in range(frames):
		s,image = Video.read()
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image = image.astype(np.uint8)
		Canny_image = cv2.Canny(image,low_threshold,high_threshold)
		out.write(Canny_image)
			
	return Canny_image

def ROI_Remove(save_file,Filename):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 20.0, (width,height),0)
	removed_image = np.zeros((height,width))

	s,frame = Video.read()
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	pl.figure()
	pl.title("Select mask")
	pl.imshow(frame, cmap=mpl_cm.Greys_r)
	pts = []
	while not len(pts):
		pts = pl.ginput(0)
	pl.close()
	path = mpl_path.Path(pts)
	mask = np.zeros((height,width), dtype=np.uint8)
	for ridx,row in enumerate(mask):
		for cidx,pt in enumerate(row):
			if path.contains_point([cidx, ridx]):
				mask[ridx,cidx] = 1
			
	frame*=mask
	frame = frame.astype(np.uint8)
	out.write(frame)
		
	for n in range(frames-1):
		
		s,frame = Video.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frame *= mask
		#idx = np.where(mask==0)
		#print idx
		frame = frame.astype(np.uint8)
		out.write(frame)
	
	Video.release()
	out.release()
	
def Threshold_IMG(save_file,Filename,threshold):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	threshold_image = np.zeros((height,width))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(save_file,FOURCC, 20.0, (width,height),0)
	for n in range(frames):
		s,image = Video.read()
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		idx = np.where(image < threshold)
		image[idx[0],idx[1]] = 0				
		out.write(image)

	out.release()
	Video.release()
	
if __name__ == "__main__":
    Main()				
	

	
	