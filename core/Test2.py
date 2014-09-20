import numpy as np
import cv2

STD_img = cv2.imread('C:\Users\Camera\Desktop\STD_DEV.png')
GRAD_img = cv2.imread('C:\Users\Camera\Desktop\Gradient.png')

image = cv2.imread('C:\Users\Camera\Desktop\GradientofMedian#.png')

STD_img = cv2.cvtColor(STD_img, cv2.COLOR_BGR2GRAY)
GRAD_img = cv2.cvtColor(GRAD_img, cv2.COLOR_BGR2GRAY)

STD_img = np.int64(STD_img)
GRAD_img = np.int64(GRAD_img)

idx = np.where(image > 10)
image[idx[0],idx[1]] = 255
cv2.imwrite('C:\Users\Camera\Desktop\BRIGHT.png', image)

# for ridx,row in enumerate(STD_img):
	# for cidx, pt in enumerate(row):
		# STD_img[ridx,cidx] = (np.power(STD_img[ridx,cidx],1)/Maximum)*255
		# STD_img[ridx,cidx] = (np.power(STD_img[ridx,cidx],0.5))
		
# Maximum = np.amax(STD_img)
# for ridx,row in enumerate(STD_img):
	# for cidx, pt in enumerate(row):
		# STD_img[ridx,cidx] = (np.power(STD_img[ridx,cidx],1)/Maximum)*255
		
# img = STD_img * GRAD_img
# maximum = np.amax(img)
# img = np.int32((np.float64(img)/(np.float64(maximum))) * 255)

#cv2.imwrite('C:\Users\Camera\Desktop\GRADANDSTD.png', img)		
#np.save('C:\Users\Camera\Desktop\STD_DEV.npy', STD_img)