# import the necessary packages
import cv2
import numpy as np
import imutils

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape

	def findLargestRectangle(self,cnts):
		if len(cnts) == 0:
			return []
		rects = []
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)
			if len(approx) == 4:
				rects.append(c)
		if(len(rects) == 0):
			return []
		return  max(rects, key = cv2.contourArea)

	def alignImage(this, image,c, cX, cY):
   # Uncomment for theta in radians
   #theta *= 180/np.pi

		center=(cX,cY)
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)



		diag1 = (int((approx[1][0][0] + abs(approx[3][0][0]))/2),
			int((approx[1][0][1] + abs(approx[3][0][1]))/2))

		diag2 = (int((approx[2][0][0] + abs(approx[0][0][0]))/2),
			int((approx[2][0][1] + abs(approx[0][0][1]))/2))

		dY = diag1[1] - diag2[1]
		dX = diag1[0] - diag2[0]
		theta =  np.degrees(np.arctan2(dY, dX)) - 180

		if theta > 90 or theta < -90:
			return image

		shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

		matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
		image = cv2.warpAffine( src=image, M=matrix, dsize=shape )
		

		return image
	
	def getContours(self, image):

		imageArea = self.__getImageArea(image)
		
		it = 0
		ih=130
		il=120
		
		while ih != 250:

			if it % 2 == 0:
				cnts, ratio, area = self.__getContours(image,il,255)
			else:
				cnts, ratio, area = self.__getContours(image,ih,255)
			if self.__isValidRect(imageArea,area,cnts) == True:
				return cnts, ratio
			else:
				if it % 2 == 0:
					il -= 10
				else:
					ih += 10
				it+=1; 
		
		it = 0
		ih=130
		il=120
		
		while ih != 250:

			if it % 2 == 0:
				cnts, ratio, area = self.__getContours(image,il,255, True)
			else:
				cnts, ratio, area = self.__getContours(image,ih,255, True)
			if self.__isValidRect(imageArea,area,cnts) == True:
				return cnts, ratio
			else:
				if it % 2 == 0:
					il -= 10
				else:
					ih += 10
				it+=1; 

		return [],ratio
	
	def __isValidRect(self,imageArea,area,cnts):
		if len(cnts) == 0:
			return False
		if(area >= imageArea*.33 and area != imageArea):
			return True
		else:
			return False

		
	def __getContours(self, image, lower,upper, invert = False):
		#resize it to a smaller factor so that
		# the shapes can be approximated better
		resized = imutils.resize(image, width=300)
		ratio = image.shape[0] / float(resized.shape[0])
 
		# convert the resized image to grayscale, blur it slightly,
		# and threshold it



		hsl = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
		Lchannel = hsl[:,:,1]
    	# define range of white color in HSV
    	# change it according to your need !
		mask = cv2.inRange(Lchannel, lower, upper)

		if(invert == True):
			mask = 255 - mask

		#gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(mask, (5, 5), 0)

		#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

		
		# find contours in the thresholded image and initialize the
		# shape detector
		cnts = cv2.findContours(blurred, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		area = 0
		r = self.findLargestRectangle(cnts)
		if len(r) == 0:
			cv2.imshow("Lower "+str(lower)+" Cnt "+str(len(cnts)), blurred)
			cv2.waitKey(250)
			cv2.destroyWindow("Lower "+str(lower)+" Cnt "+str(len(cnts)))	
		else:
			area = cv2.contourArea(r)
			cv2.drawContours(blurred, r, -1, (255,0,0), 10)
			cv2.imshow("Lower "+str(lower)+" Cnt "+str(len(cnts))+" Area "+str(area), blurred)
			cv2.waitKey(250)
			cv2.destroyWindow("Lower "+str(lower)+" Cnt "+str(len(cnts))+" Area "+str(area))

		return cnts,ratio, area


	def __getImageArea(self, image):
		resized = imutils.resize(image, width=300)
		ratio = image.shape[0] / float(resized.shape[0])
		hsl = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
		Lchannel = hsl[:,:,1]
    	# define range of white color in HSV
    	# change it according to your need !
		mask = cv2.inRange(Lchannel, 0, 255)

		blurred = cv2.GaussianBlur(mask, (5, 5), 0)

		cnts = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		r = self.findLargestRectangle(cnts)

		return cv2.contourArea(r)


	def computerContour(self, c, ratio):
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")

		return c, cX,cY

	def findPoints(self,arr):
		x = []
		y = []
		
		for p in arr:
			x.append(p[0][0])
			y.append(p[0][1])
		x.sort()
		y.sort()
		return  x,y



	def orderPoints(self,arr):
		a= []
		b= []
		c= []
		d= []
		x,y  = self.findPoints(arr)
		
		for p in arr:
			if (p[0][0] == x[0] or  p[0][0] == x[1]) and (p[0][1] == y[2] or p[0][1] == y[3]):
				a = p
			elif (p[0][0] == x[2] or  p[0][0] == x[3]) and (p[0][1] == y[2] or p[0][1] == y[3]):
				b = p
			elif (p[0][0] == x[0] or  p[0][0] == x[1]) and (p[0][1] == y[0] or p[0][1] == y[1]):
				c = p
			else:
				d = p
		pass

		return [a,b,c,d]

	