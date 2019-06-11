

# import the necessary packages

import numpy as np
import cv2
from collections import OrderedDict
import argparse
import imutils
import dlib
import cv2






class FaceAligner:
    def __init__(self):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = (0.35, 0.35);
        self.desiredFaceWidth = 400
        self.desiredFaceHeight = 400

        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat");

        self.detector = dlib.get_frontal_face_detector()


        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth


    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
   	    ("right_eyebrow", (17, 22)),
   	    ("left_eyebrow", (22, 27)),
   	    ("right_eye", (36, 42)),
   	    ("left_eye", (42, 48)),
   	    ("nose", (27, 36)),
   	    ("jaw", (0, 17))
        ])

    def rect_to_bb(self,rect):
	    # take a bounding predicted by dlib and convert it
	    # to the format (x, y, w, h) as we would normally do
	    # with OpenCV
	    x = rect.left()
	    y = rect.top()
	    w = rect.right() - x
	    h = rect.bottom() - y

	    # return a tuple of (x, y, w, h)
	    return (x, y, w, h)

    def shape_to_np(self,shape, dtype="int"):
	    # initialize the list of (x, y)-coordinates
	    coords = np.zeros((68, 2), dtype=dtype)

	    # loop over the 68 facial landmarks and convert them
	    # to a 2-tuple of (x, y)-coordinates
	    for i in range(0, 68):
		    coords[i] = (shape.part(i).x, shape.part(i).y)

	    # return the list of (x, y)-coordinates
	    return coords

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = self.shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = self.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

        
    def align_face(self, raw_image):

        # load the input image, resize it, and convert it to grayscale
        image = imutils.resize(raw_image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # show the original input image and detect faces in the grayscale
        # image
        rects = self.detector(gray, 2)

        # loop over the face detections
        if len(rects) == 0:
            return None

        faceAligns = []
	    # extract the ROI of the *original* face, then align the face
	    # using facial landmarks
        for r in rects:
            (  x, y, w, h) = self.rect_to_bb(r)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligns.append(self.align(image, gray, r))
        
        return faceAligns



