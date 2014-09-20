import numpy as np
import cv2
import cv2.cv as cv
import os
import sys
import time
import pylab as pl
import matplotlib.cm as mpl_cm
import time as pytime
from matplotlib import path as mpl_path
from core.DenseLimbTracker import Main
from core.DenseFlowROI import Main
from core.OpticalFlow import Main
import core.EditVideo
from core.HeatPlot import Main
from core.MotorPrimitives_HeatPlot import Main

def Main_Init():

#####################################################
#                PARAMETERS:                        #
#  FOR MORE INFORMATION ON HOW TO MANIPULATE THESE  #
#  PARAMETERS, CONSULE THE README.TXT FILE IN THIS  #
#  DIRECTORY.                                       #
#####################################################
#  FOR USE IN ANALYSIS OF ONE PHOTON AND TWO PHOTON #
#  IMAGES OF MICE AND THEIR RESPECTIVE MOTOR        #
#  PRIMITIVES                                       #
#####################################################
#  IF YOU WOULD LIKE TO APPLY AN INDIVIDUAL         #
#  OPERATION CONTAINED IN THIS ANALYSIS, THEN YOU   #
#  SHOULD RUN THE SCRIPTS INDIVIDUALLY.             #
#####################################################

# BELOW IS USER DEFINED VARIABLES


	#For best results, ensure the directories are empty before running analysis. Make sure to use \\ not \
	Directory_of_Parent_Video = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG051514-01\\21072014\\Trial1"
	Directory_to_save_analysis_ROOT = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\AG051514-01\\21072014\\Trial1\\Analysis"


	#If you would like to image only one video then have both numbers be the number of that video
	Videos_to_be_Imaged = (1,97)

	Median_Filter_Kernel_Size = 7
	
	#Allows the user to choose the ROI defining motor primitives every video
	Select_ROI_Every_Video = False #Recommended False
	
	Create_Masked_Video = True #Both are fine
	
	#Takes the mean of the median angle when determining the angle of the motor primitive.
	#Otherwise the user manually inputs the value
	Automatically_Calculate_Projection_Angle = True #Recommended True for better analysis
	
	Display_Images = False
	
	Number_of_Motor_Primitive_Masks = 4
		
	#The file extension is not necessary
	Name_of_Median_Filtered_Video = "Median_Video"
	Name_of_Gradient_Median_Filtered_Video = "Gradient_Video"
	Name_of_Masked_Gradient_Median_Filtered_Video = "Video_to_be_Analyzed"

# END OF USER DEFINED VARIABLES. MANIPULATION OF CODE BELOW FOR ADVANCED USERS ONLY
	
	if not os.path.isdir(Directory_to_save_analysis_ROOT):
		os.mkdir(Directory_to_save_analysis_ROOT)
	# MAIN LOOP
	
	print "Beginning Analysis..."
	print " "
	
	for n in range(Videos_to_be_Imaged[0],Videos_to_be_Imaged[1]+1):
		print "Analyzing Video %i"%n
		Directory_to_save_analysis = os.path.join(Directory_to_save_analysis_ROOT,"Video %i"%n)
		if not os.path.isdir(Directory_to_save_analysis):
			os.mkdir(Directory_to_save_analysis)
		median_savename,gradient_savename,final_savename = Set_Save_Directory(Directory_to_save_analysis,Name_of_Median_Filtered_Video,Name_of_Gradient_Median_Filtered_Video,Name_of_Masked_Gradient_Median_Filtered_Video)
		MTR_PRIM_mask_savename = os.path.join(Directory_to_save_analysis,"Motor Primitives Masks.png")
		WHEEL_mask_savename = os.path.join(Directory_to_save_analysis,"Wheel Mask.png")
		
		file = os.path.join(Directory_of_Parent_Video,"PS3_Vid%i.avi"%n)
		
		core.EditVideo.Median_Filter(Median_Filter_Kernel_Size,file,median_savename)
		print "Median Filter Complete"
		core.EditVideo.Gradient_Filter(gradient_savename,median_savename)
		print "Gradient Complete"
		
		#Format of Masks matrix is [n,y,x], where n is the number of masks
		if (Select_ROI_Every_Video or n==Videos_to_be_Imaged[0]):
			print "Please select the mask(s)"
			Masks = Set_Masks(gradient_savename,Number_of_Motor_Primitive_Masks,"Select Motor Primitive Region",MTR_PRIM_mask_savename)
			
		if Create_Masked_Video:
			Removal_Filter(gradient_savename,final_savename,Masks) #Not actually Necessary
			print "Removed Videos Finished"
			
		core.EditVideo.Grid(Directory_of_Parent_Video,"PG_Vid%i"%n, Directory_to_save_analysis, (160,160))
		print "Finished Drawing Grid Images"
			
		
		
		core.HeatPlot.Main(Directory_of_Parent_Video,(n,n),Directory_to_save_analysis,Display_Images)
		print "Heat Plot Complete"
				
		core.DenseFlowROI.Main(gradient_savename,Directory_to_save_analysis,Automatically_Calculate_Projection_Angle,Masks,Display_Images = False)
		print "Motor Primitives Complete"
		
		core.MotorPrimitives_HeatPlot.Main(Directory_to_save_analysis,Display_Images)
		
		#if (Select_ROI_Every_Video or n==Videos_to_be_Imaged[0]):
		#	Wheel_Mask = Set_Masks(gradient_savename,1,"Select Wheel Where No Part of the Mouse Touches Ever",WHEEL_mask_savename)[0]
		#core.OpticalFlow.Main(Directory_to_save_analysis,gradient_savename,Wheel_Mask,Display_Images)
		#print "Wheel Angular Momentum Plot Complete"
			
			
def Removal_Filter(Filename,destination,Masks):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(destination,FOURCC, 20.0, (width,height),0)
	
	for m in range(frames):
	
		s,frame = Video.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frame = frame.astype(np.uint8)
		
		tmasks = np.zeros(np.shape(frame))
		for n in range(np.shape(Masks)[0]):
			tmasks += Masks[n]
		tmasks = (tmasks>0).astype(np.uint8)
		frame*=tmasks
		
		out.write(frame)
		
	Video.release()
	out.release()
	
def Set_Masks(Filename,number,message,save_name):
	Video = cv2.VideoCapture(Filename)
	height = int(Video.get(4))
	width = int(Video.get(3))
	frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
	FOURCC = cv2.cv.CV_FOURCC(*'XVID')
	color_names_RBG = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0)] #BGR
	
	s,frame = Video.read()
	disp_frame = frame[:]
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	Masks = np.zeros((number,height,width),dtype=np.uint8)
	
	for n in range(number):
		pl.figure()
		pl.title(message+" %i"%(n+1))
		pl.imshow(disp_frame, cmap=mpl_cm.Greys_r)
		pts = []
		while not len(pts):
			pts = pl.ginput(0)
		pts = np.array(pts, dtype=np.int32)
		pl.close()
		path = mpl_path.Path(pts)
		for ridx,row in enumerate(frame):
			for cidx,pt in enumerate(row):
				if path.contains_point([cidx, ridx]):
					Masks[n,ridx,cidx] = 1
					
		cv2.polylines(disp_frame, [pts], 1, color_names_RBG[n%6], thickness=1) #BGR
		
	cv2.imwrite(save_name,disp_frame)
	Video.release()

	return Masks

def Set_Save_Directory(Directory_to_save_analysis,Name_of_Median_Filtered_Video,Name_of_Gradient_Median_Filtered_Video,Name_of_Masked_Gradient_Median_Filtered_Video):
	if Name_of_Median_Filtered_Video[-4:] == ".avi":
		median_savename = os.path.join(Directory_to_save_analysis,Name_of_Median_Filtered_Video)
	else:
		median_savename = os.path.join(Directory_to_save_analysis,Name_of_Median_Filtered_Video+".avi")
	
	if Name_of_Gradient_Median_Filtered_Video[-4:] == ".avi":
		gradient_savename = os.path.join(Directory_to_save_analysis,Name_of_Gradient_Median_Filtered_Video)
	else:
		gradient_savename = os.path.join(Directory_to_save_analysis,Name_of_Gradient_Median_Filtered_Video+".avi")
	
	if Name_of_Masked_Gradient_Median_Filtered_Video[-4:] == ".avi":
		final_savename = os.path.join(Directory_to_save_analysis,Name_of_Masked_Gradient_Median_Filtered_Video)
	else:
		final_savename = os.path.join(Directory_to_save_analysis,Name_of_Masked_Gradient_Median_Filtered_Video+".avi")
	
	return median_savename,gradient_savename,final_savename
	
if __name__ == "__main__":
    Main_Init()	