import cv2 as cv
import imutils

image = cv.imread("tetris_blocks.png")
(h , w, d)  = image.shape
print(h, w , d) #height,width and color channel
(b, g, r) = image[100,50]
print(b, g, r)
#Array Slicing
roi = image[60:160,320:420]
#Resizing image
#r = 300.0 / w
#dim = (500, int(h * r))
#Resizing image using imutils instead of calculating aspect ratio
resize = imutils.resize(image, width = 300)
#Rotate an image 
#center = (w//2, h//2)
#M = cv.getRotationMatrix2D(center,45,1.0)
#rotated = cv.warpAffine(image,M,(w, h))
#rotated = imutils.rotate(image,-45)

#Bluring an image
#blur = cv.GaussianBlur(image,(7,7),0)

#draw a red rectangle around the face
#image1 = image.copy()
#cv.rectangle(image1,(320,60),(420,160),(0,0,255),2)

#draw a red circle
#image1 = image.copy()
#cv.circle(image1,(300,150),20,(0,0,255),-1)

#draw green text on the image
#image1 = image.copy()
#cv.putText(image1,"OpenCV",(10,25),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

#convert BGR to grayscale
#gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#Edge detection using canny edge
edge = cv.Canny(image,30,150)

cv.imshow('Blurred Image',edge)
cv.imshow('original',image)
cv.waitKey(0)
cv.destroyAllWindows()