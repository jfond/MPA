import cv2
import numpy as np
import cv2.cv as cv
import time

#Video = cv2.VideoCapture('C:\\Users\\Camera\\Desktop\\Edge.avi')
#s, image = Video.read(0)
image = cv2.imread('C:\\Users\\Camera\\Desktop\\Canny.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
idx = np.where(image < 128)
image[idx[0],idx[1]] = 0
idx = np.where(image>=128)
image[idx[0],idx[1]] = 255

dimensions = np.shape(image)
width = dimensions[1]
height = dimensions[0]

trace_img = np.zeros((dimensions[0],dimensions[1],3))

trace_img[:,:,1] = image

idx = np.where(image == np.amax(image))
y = idx[0][0]
x = idx[1][0]

y =156
x = 286

image[y,x] = 0
trace_img[y,x] = [0,0,255]
Border_Points = [[y],[x]]

search_dist = 5
jump_dist = 5

list = np.zeros(((search_dist*2+1),(search_dist*2+1)))
for tx in range(-search_dist,search_dist+1):
	for ty in range(-search_dist,search_dist+1):
		if (y + ty >= 0) and (y + ty < height) and (x + tx >= 0) and (x + tx < width):
			list[ty + search_dist,tx+search_dist] = image[y+ty,x+tx]



idx = np.where(list == 255)
dy = y - Border_Points[0][-1] + int(1*(idx[0][-1] - search_dist))
dx = x - Border_Points[1][-1] + int(1*(idx[1][-1] - search_dist))
erase_dist = int(np.sqrt(np.power(dx,2)+np.power(dy,2))) - 1
if erase_dist < 0:
	erase_dist = 0	
image[Border_Points[0][-1]-erase_dist:Border_Points[0][-1]+erase_dist+1,Border_Points[1][-1]-erase_dist:Border_Points[1][-1]+erase_dist+1] = 0	
y = Border_Points[0][-1] + dy
x = Border_Points[1][-1] + dx
search_dist = int(np.sqrt(np.power(dx,2)+np.power(dy,2))) + 1
Border_Points[0].append(y)
Border_Points[1].append(x)
trace_img[y,x] = [0,0,255]
if (y + dy >= 0) and (y + dy < height): 
	y+=dy
if(x + dx >= 0) and (x + dx < width):
	x+=dx


for n in range(100):
	print n
	list = np.zeros(((search_dist*2+1),(search_dist*2+1)))
	for tx in range(-search_dist,search_dist+1):
		for ty in range(-search_dist,search_dist+1):
			if (y + ty >= 0) and (y + ty < height) and (x + tx >= 0) and (x + tx < width):
				#list[ty + search_dist,tx+search_dist] = ((np.abs(image[y+ty,x+tx] - past_value) * (2 - int(image[y+ty,x+tx] >= past_value))))
				if image[y+ty,x+tx] == 255:
					list[ty + search_dist,tx+search_dist] = np.exp(np.power((np.sqrt(np.power(np.float32(y+ty-Border_Points[0][-1]),2)+np.power(np.float32(x+tx-Border_Points[1][-1]),2)) - jump_dist),2) * -1 / 2)
					#list[ty + search_dist,tx+search_dist] = 1
				elif image[y+ty,x+tx] == 128:
					list[ty + search_dist,tx+search_dist] = np.exp(np.power((np.sqrt(np.power(np.float32(y+ty-Border_Points[0][-1]),2)+np.power(np.float32(x+tx-Border_Points[1][-1]),2)) - jump_dist),2) * -1 / 2)/4
				else:
					list[ty + search_dist,tx+search_dist] = 2048 #So it isnt a minimum
			else:
				list[ty + search_dist,tx+search_dist] = 2048 #So it isnt a minimum
	
	exit = np.where(list != 0)
	if len(exit[0]) == 0:
		break	
	idx = np.where(list != 2048)
	idx = [idx[0].astype(np.float16),idx[1].astype(np.float16)]
	#print idx
	#print len(idx[0])
	
	if len(idx[0]) > 0.9:
		sum = [0,0]
		norm = 0
		for c in range(len(idx[0])):
			weight = list[idx[0][c],idx[1][c]]
			sum[0] += weight * idx[0][c]
			sum[1] += weight * idx[1][c]
			norm += weight

		idx_x = int(sum[1]/norm)
		idx_y = int(sum[0]/norm)
		#idx_x = int(np.mean(idx[1]))
		#idx_y = int(np.mean(idx[0]))
		idx = [[idx_y],[idx_x]]
		
	else:
		y += dy
		x += dx
		
		list = np.zeros(((search_dist*4+1),(search_dist*4+1)))
		for tx in range(-search_dist,search_dist+1):
			for ty in range(-search_dist,search_dist+1):
				if (y + ty >= 0) and (y + ty < height) and (x + tx >= 0) and (x + tx < width):
					#list[ty + search_dist,tx+search_dist] = ((np.abs(image[y+ty,x+tx] - past_value) * (2 - int(image[y+ty,x+tx] >= past_value))))
					if image[y+ty,x+tx] == 255:
						list[ty + search_dist,tx+search_dist] = np.exp(np.power((np.sqrt(np.power(y+ty-Border_Points[0][-1],2)+np.power(x+tx-Border_Points[1][-1],2)) - (jump_dist*3)),2) * -1 / 2)
						#list[ty + search_dist,tx+search_dist] = 1
					elif image[y+ty,x+tx] == 128:
						list[ty + search_dist,tx+search_dist] = np.exp(np.power((np.sqrt(np.power(y+ty-Border_Points[0][-1],2)+np.power(x+tx-Border_Points[1][-1],2)) - (jump_dist*3)),2) * -1 / 2)/4
					else:
						list[ty + search_dist,tx+search_dist] = 2048 #So it isnt a minimum
				else:
					list[ty + search_dist,tx+search_dist] = 2048 #So it isnt a minimum
		
		exit = np.where(list != 0)
		if len(exit[0]) == 0:
			break	
		idx = np.where(list != 2048)
		idx = [idx[0].astype(np.float16),idx[1].astype(np.float16)]
		sum = [0,0]
		norm = 0
		for c in range(len(idx[0])):
			weight = list[idx[0][c],idx[1][c]]
			sum[0] += weight * idx[0][c]
			sum[1] += weight * idx[1][c]
			norm += weight

		print list
		trace_img[y-search_dist,y+search_dist+1,x-search_dist,x+search_dist+1] = [0,0,255]
			
		idx_x = int(sum[1]/norm)
		idx_y = int(sum[0]/norm)
		#idx_x = int(np.mean(idx[1]))
		#idx_y = int(np.mean(idx[0]))
		idx = [[idx_y],[idx_x]]	
		

	dy = y - Border_Points[0][-1] + int(1*(idx[0][0] - search_dist))
	dx = x - Border_Points[1][-1] + int(1*(idx[1][0] - search_dist))
	erase_dist = int(np.sqrt(np.power(dx,2)+np.power(dy,2))) - 1
	if erase_dist < 0:
		erase_dist = 0	
	image[Border_Points[0][-1]-erase_dist:Border_Points[0][-1]+erase_dist+1,Border_Points[1][-1]-erase_dist:Border_Points[1][-1]+erase_dist+1] = 0	
	y = Border_Points[0][-1] + dy
	x = Border_Points[1][-1] + dx

	#dx = int(1*(idx[1][0] - search_dist))
	#dy = int(1*(idx[0][0] - search_dist))
	
	search_dist = int(np.sqrt(np.power(dx,2)+np.power(dy,2))) + 1
		
	Border_Points[0].append(y)
	Border_Points[1].append(x)


	cv2.imwrite('C:\Users\Camera\Desktop\Trace.png', trace_img)
	trace_img[y,x] = [0,0,255]
	if (y + dy >= 0) and (y + dy < height): 
		y+=dy
	if(x + dx >= 0) and (x + dx < width):
		x+=dx
	#cv2.imwrite('C:\Users\Camera\Desktop\Trace.png', trace_img)
	time.sleep(1)

cv2.imwrite('C:\Users\Camera\Desktop\Trace.png', trace_img)
