import shapedetector
import argparse
import imutils
import cv2
import numpy as np
 
# construct the argument parse and parse the arguments

sd = shapedetector.ShapeDetector()

def getLargestRectangle(imge):
	cnts, ratio = sd.getContours(imge)
	c = sd.findLargestRectangle(cnts)
	c, cX, cY = sd.computerContour(c,ratio)
	
	return c, cX, cY

	# load the image and 

def processBusinessCard(image):
	
	cv2.imshow("Image", image)
	cv2.waitKey(2000)
	cv2.destroyWindow("Image")

	

	c, cX, cY = getLargestRectangle(image)

	center=(cX,cY)
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	approx = sd.orderPoints(approx)


	width = 300
	height = 150

	if approx[1][0][1] * 1.10 > approx[1][0][0]:
		temp = width
		width = height
		height = temp


	pts1 = np.float32([
		[approx[0][0][0],approx[0][0][1]],
		[approx[1][0][0],approx[1][0][1]],
		[approx[2][0][0],approx[2][0][1]],
		[approx[3][0][0],approx[3][0][1]]])
	pts2 = np.float32([[0,height],[width,height],[0,0],[width,0]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(image,M,(width,height))


	xOffeset = int((width*.12)/2)
	yOffeset = int((height*.12)/2)

	xLength = int(width*.95)
	yLength = int(height*.95)

	ffinal = dst[yOffeset:yLength, xOffeset:xLength]

	cv2.imwrite("BusinessCard.jpeg", ffinal)

	cv2.imshow("Image", ffinal)
	cv2.waitKey(2000)
	cv2.destroyWindow("Image")

image = imutils.resize(cv2.imread("./darktest3.jpg"), width=720)
processBusinessCard(image)
image = imutils.resize(cv2.imread("./image1.jpeg"), width=720)
processBusinessCard(image)
image = imutils.resize(cv2.imread("./darktest2.jpg"), width=720)
processBusinessCard(image)