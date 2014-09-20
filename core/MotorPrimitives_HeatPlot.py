import cv2
import numpy as np
import cv2.cv as cv
import time
import matplotlib.pyplot as plt
import os
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import time
import os
from scipy.signal import medfilt

def Main(folder_location,Display_Images):


	Motor_primitives = np.load(os.path.join(folder_location,"Motor_Primitives_Data.npy"))
	median_angle = Motor_primitives[:,:,0]
	mag_list = Motor_primitives[:,:,1]
	masks = np.shape(Motor_primitives)[0]
	
	interspace_dist = 5

	line_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	#myplot = plt.plot(np.sign(median_angle)*mag_list,'r')
	
	#im = plt.imshow(mean_plot_vals, interpolation='nearest',aspect='auto')
	
	plotme = np.sign(median_angle)*mag_list
	#plotme /= np.amax(np.abs(plotme))
	#print np.amax(np.abs(plotme))
	for n in range(masks):
		plotme[n,:] /= np.amax(np.abs(plotme[n,:]))
	frames = len(plotme[0,:])
	
	range_plot_vals = np.zeros((masks, 2))	
	for n in range(masks): #Stuff for line plot rather than heat plot
		range_plot_vals[n,0] = range_plot_vals[n-1,0] + np.amin(plotme[n,:])
		range_plot_vals[n,1] = range_plot_vals[n-1,1] + np.amax(plotme[n,:])
		plotme[n,:] += range_plot_vals[n-1,1] - range_plot_vals[n,0] + interspace_dist

		#plot = cv2.GaussianBlur(plotme[n,:],(0,time_blur_range),time_blur_strength) #Smooth the average in the time dimension
		plt.plot(range(1,frames+1), plotme[n,:], line_colors[n%7])	
		
	#plt.imshow(plotme, interpolation='nearest',aspect='auto', extent = [0,4,np.shape(Motor_primitives)[0],1])

	plt.xlabel('Time (Seconds)')
	plt.ylabel('Motor Primitive')

		

	#plt.jet()
	#cb = plt.colorbar()
	#cb.set_ticks([np.amin(plotme), np.mean((np.amin(plotme),np.amax(plotme))), np.amax(plotme)])  # force there to be only 3 ticks
	#cb.set_ticklabels([str(np.amin(plotme))[:5], 'Normalized Velocity', str(np.amax(plotme))[:5]])  # put text labels on them


	plt.subplots_adjust(left=0.15)
	plt.savefig(os.path.join(folder_location,"Motor Primitives Separate Lines.png"))
		
	if Display_Images:
		plt.show()
		
	plt.clf()

    
    
    
    
if __name__ == "__main__":
	
	folder_location = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\18072014\\Trial1\\Analysis\\Video 34"
	Display_Images = True
	
	Main(folder_location,Display_Images)        