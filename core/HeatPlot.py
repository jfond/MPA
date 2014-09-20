import cv2
import numpy as np
import cv2.cv as cv
import time
import matplotlib.pyplot as plt
import os

def Main(trial_location,video_range,save_location,Display_Images):


	square_size = 160 #side length of each square    

	time_blur_range = 3
	time_blur_strength = 3
	spatial_blur_range = 5
	spatial_blur_strength = 0.65

	brain_location = [[0,960],[0,1280]]
	mouse_location = [[0,240],[1281,1600]]

	subtract_percentile = False
	line_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if type(video_range[0]) != int or type(video_range[1]) != int:
		raise Exception("Video range not valid")

	brain_height = brain_location[0][1] - brain_location[0][0]
	brain_width = brain_location[1][1] - brain_location[1][0]

	mouse_width = mouse_location[0][1] = mouse_location[0][0]
	mouse_height= mouse_location[1][1] = mouse_location[1][0]	
	
	for num in range(video_range[0],video_range[1]+1):
		Video = cv2.VideoCapture(os.path.join(trial_location,"Synchronized_Vid%i.avi"%num))

		height_total = int(Video.get(4))
		width_total = int(Video.get(3))
		fps = int(Video.get(cv.CV_CAP_PROP_FPS))
		
		frames = int(Video.get(cv.CV_CAP_PROP_FRAME_COUNT))
		num_squares_x = int(brain_width/square_size)
		num_squares_y = int(brain_height/square_size)

		mean_plot_vals = np.zeros((num_squares_y*num_squares_x, frames))
		mean_plot_vals_no_range = np.zeros((num_squares_y*num_squares_x, frames))
		range_plot_vals = np.zeros((num_squares_y*num_squares_x, 2))
		for t in range(frames):
			s,image = Video.read()
#           image = cv2.GaussianBlur(image,(spatial_blur_range,spatial_blur_range),spatial_blur_strength) #Gaussian Smoothing filter in the x,y spatial dimensions
            
#            if subtract_percentile:
#                percentile = np.percentile(image,8)
#                image -= percentile
            
			for y in range(num_squares_y):
				for x in range(num_squares_x):
					avg = np.mean(image[y*square_size:(y+1)*square_size, x*square_size:(x+1)*square_size])                                        
					mean_plot_vals[(y*num_squares_x)+x,t] = avg

		for n in range(num_squares_x*num_squares_y):   
			F=np.percentile(mean_plot_vals[n,:],8);
			if F > 0:
				mean_plot_vals[n,:]=(mean_plot_vals[n,:]-F)/F
			else:
				text_file = open(os.path.join(save_location,'Notes.txt'), "a")
				text_file.write("The 8th percentile was 0 or less than 0, so dF/F could not be properly calculated for region %i \n"%n)
				text_file.close()
				mean_plot_vals[n,:]=(mean_plot_vals[n,:]-F)
		
		
		fig, ax = plt.subplots(1,1,figsize=(16, 12), dpi=100, subplot_kw={'aspect': 'equal'})

		plt.imshow(mean_plot_vals, interpolation='nearest',aspect='auto',extent=[0,frames/fps,48,1])
		np.save(os.path.join(save_location,'Neural Data.npy'),mean_plot_vals) 
  
		#for n in range(num_squares_x*num_squares_y): #Stuff for line plot rather than heat plot
#                range_plot_vals[n,0] = range_plot_vals[n-1,0] + np.amin(mean_plot_vals[n,:])
#                range_plot_vals[n,1] = range_plot_vals[n-1,1] + np.amax(mean_plot_vals[n,:])
#                mean_plot_vals[n,:] += range_plot_vals[n-1,1] - range_plot_vals[n,0] + interspace_dist
#
#                plot = cv2.GaussianBlur(mean_plot_vals[n,:],(0,time_blur_range),time_blur_strength) #Smooth the average in the time dimension
#                plt.plot(range(1,frames+1), plot, line_colors[n%7])
                    
		plt.xlabel('Time (Seconds)')
		plt.ylabel('Frame')
		#plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

		# Tweak spacing to prevent clipping of ylabel
		

		plt.jet()
		cb = plt.colorbar()
		cb.set_ticks([np.amin(mean_plot_vals), np.mean((np.amin(mean_plot_vals),np.amax(mean_plot_vals))), np.amax(mean_plot_vals)])  # force there to be only 3 ticks
		cb.set_ticklabels([str(np.amin(mean_plot_vals))[:5], 'dF/F Magnitude', str(np.amax(mean_plot_vals))[:5]])  # put text labels on them

		
		plt.subplots_adjust(left=0.15)
		plt.savefig(os.path.join(save_location,"Heat Plot.png"))
		
		if Display_Images:
			plt.show()
		
		plt.clf()

    
    
    
    
if __name__ == "__main__":
	trial_location = "C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\18072014\\Trial1"
	video_range = (64,64) #Make both the same number if you want to image one video
	save_location = "C:\\Users\\Camera\\Desktop\\Video_Editing_Tools\\Analysis\\MOUSE 2014-01, NUMBER 64"
	Display_Images = True
	
	Main(trial_location,video_range,save_location,Display_Images)        