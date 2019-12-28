Document Scanner or bird-eye view is  a feature to capture the text in the image and display it in the front of the user giving it 4 point perspective transform.

# 3 steps to accomplished this task are:

1. First we need to detect the edges of the given image. We can apply any edge detecting algorithm but here I used Canny Edge Detection Algorithm.
2. Used that edges that we found to find the countours or outline representing the piece of paper being detected.
3. Apply the perspective transform to obtain the top-down view of the documents.

## Tools Required
1. OpenCV(of course)
2. skimage.filters to obtain the black and white feel of the image
3. imutils

# Results

 | Edge Detection  |  Finding Contours | Top-Down View |
| ------------- | ------------- | ------------- |
| ![Alt text](results/edge.png?raw=true "Canny Edge" )  |   ![Alt text](results/contour.png?raw=true " Contours" )  |   ![Alt text](results/Scan.png?raw=true " 4 point perspective" )
