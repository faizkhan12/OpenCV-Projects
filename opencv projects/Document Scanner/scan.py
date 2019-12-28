import cv2 as cv
import numpy as np
from skimage.filters import threshold_local
import imutils
from Transform.transform import four_point_transform

image = cv.imread('receipt.jpg')
#Height of the image
ratio = image.shape[0] / 500.0
#print(ratio)
image1 = image.copy()   #make a copy
image = imutils.resize(image,height = 500)  #resize the original image
'''Step 1 - Edge detection
convert image to grayscale, then smooth the image
'''
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5) , 0)
# now apply canny edge detection 
edged = cv.Canny(gray,75,200)

''' Step 2 - Find the Contours
'''
#Find the contour in the edges image
cnt = cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
cnt = imutils.grab_contours(cnt) #return contours differently
cnt = sorted(cnt,key = cv.contourArea, reverse = True)[:5]  # sort the contours
for i in cnt:   #loop over the contour
    peri = cv.arcLength(i, True)    #approximate the contour
    approx = cv.approxPolyDP(i , 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break
cv.drawContours(image, [screenCnt], -1, (0,255,0), 2)

'''Apply the perspective Transform
'''
warped = four_point_transform(image1,screenCnt.reshape(4 ,2) * ratio)
warped = cv.cvtColor(warped,cv.COLOR_BGR2GRAY)
thres = threshold_local(warped, 11, offset = 10, method = 'gaussian')
warped = (warped > thres).astype("uint8") * 255
image1 = imutils.resize(image1, height = 600)
warped = imutils.resize(warped, height = 600)
cv.imshow('Original',image1)
cv.imshow('Scanned',edged)
cv.waitKey(0)
cv.destroyAllWindows()
