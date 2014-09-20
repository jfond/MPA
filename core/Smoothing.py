import numpy as np
import cv2
import cv2.cv as cv

cap = cv2.VideoCapture('C:\\Users\\Camera\\Desktop\\GtHUb\\Two-Cameras\\Data\\AG052014-01\\21072014\\Trial5\\PS3_Vid83.avi')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
A = int(cap.get(3))
B = int(cap.get(4))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))

frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

t = 8

s,image = cap.read()

dist = 3
Smoothed_img = np.zeros(np.shape(image))
image = cv2.imread('C:\\Users\\Camera\\Desktop\\GradientofMedian#.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# idx = np.where(image > 156)
# image[idx[0],idx[1]] = 255
# cv2.imwrite('C:\Users\Camera\Desktop\Smooth.png', image)

#cv2.imwrite('C:\Users\Camera\Desktop\Smooth.png', image)

#image = image*image

# Maximum = np.amax(image)
# print Maximum
# for ridx,row in enumerate(image):
	# for cidx,pt in enumerate(row):
		# image[ridx,cidx] = int((float(image[ridx,cidx])/float(Maximum)) * 255)
		
# cv2.imwrite('C:\Users\Camera\Desktop\Smooth.png', image)

for ridx,row in enumerate(image):
	for cidx,pt in enumerate(row):
		sum = 0
		norm = 0
		list = []
		for x in range(-dist,dist+1):
			for y in range(-dist,dist+1):
				if (ridx + y >= 0) and (ridx + y < B) and (cidx + x >= 0) and (cidx + x < A):
					list.append(image[ridx+y,cidx+x])
					displacment = np.sqrt((x*x)+(y*y))
					weight = np.exp(dist/(2*t*t))
					norm += weight
					sum += (weight * image[ridx+y,cidx+x])
					Smoothed_img[ridx,cidx] = sum/norm
					#Smoothed_img[ridx,cidx] = min(list)
	

threshold = 32
idx_above = np.where(Smoothed_img >= threshold)
idx_below = np.where(Smoothed_img < threshold)
maxi = np.amax(Smoothed_img)
Smoothed_img[idx_below[0],idx_below[1]] = 0
Smoothed_img[idx_above[0], idx_above[1]] = np.int16(((Smoothed_img[idx_above[0], idx_above[1]])-threshold) * 256/(maxi-threshold))

cv2.imwrite('C:\Users\Camera\Desktop\Smooth.png', Smoothed_img)
cap.release()