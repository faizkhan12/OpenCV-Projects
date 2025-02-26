from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#function to compute midpoint between 2 coordinates x and y
width = 0.955
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5 )

# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())
'''
#load the image,convert into grayscale and blur it
#image = cv2.imread(args["image"])
image = cv2.imread("images\example_03.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(image,(7,7),0)

#perform edge detection, then dilation and erosion to close gaps between object edges
edged = cv2.Canny(gray,50,100)
edged = cv2.dilate(edged,None,iterations=1)
edged = cv2.erode(edged,None,iterations=1)
cnts = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

#Loops over the contours
for c in cnts:
	#if contour is not long, then simply ignore it
	if(cv2.contourArea(c) < 100):
		continue
	#else if contour is large enough then compute the rotating bounding box of the contour
	image1 = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box,dtype='int')

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(image1, [box.astype("int")], -1, (0,255,0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(image1, (int(x), int(y)), 5, (0, 0, 255), -1)
	#unpack the ordered bounding box 
	(tl, tr, br , bl) = box
	#compute the midpoint between top-left and top-right 
	(tltrX, tltrY) = midpoint(tl,tr)
	#Compute the midpoint beetween bottom-left and bottom-right
	(blbrX, blbrY) = midpoint(bl, br)
	#Also compute the midpoint b/w top-left and bottom-left
	(tlblX, tlblY) = midpoint(tl,bl)
	#Also compute the midpoint b/w top-right and bottom-right
	(trbrX, trbrY) = midpoint(tr,br)

	# draw the midpoints on the image
	cv2.circle(image1, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(image1, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(image1,(int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(image1, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(image1, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(image1, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / width
	    
        # compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
 
	# draw the object sizes on the image
	cv2.putText(image1, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(image1, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
 
	# show the output image
	cv2.imshow("Image", image1)
	cv2.waitKey(0)

#cv2.imshow('window',image1)
#cv2.destroyAllWindows
