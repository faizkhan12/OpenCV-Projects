import cv2 as cv
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
import time

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# Scale and load model
print("[INFO] loading model.....")
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialise the video stream
print("[INFO] starting video stream....")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame
    frame = vs.read()
    # resize it to have a maximum width of 400 pixels
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions
    (h, w) = frame.shape[:2]
    # Now convert it to a blob
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300,300)),1.0,(300, 300), (104.0, 177.0, 123.0))

    # pass the blob throught the n/w and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv.putText(frame, text, (startX, y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv.destroyAllWindows()
vs.stop()
