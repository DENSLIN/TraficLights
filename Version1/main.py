import numpy as np
# import argparse
# import imutils
import time
import cv2
# import os
# from tqdm import tqdm
from collections import Counter

# load the COCO class labels our YOLO model was trained on
labelsPath = "/home/denslin/ML/trafic lights/car detection 2/yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "/home/denslin/ML/trafic lights/car detection 2/yolo-coco/yolov3.weights"
configPath = "/home/denslin/ML/trafic lights/car detection 2/yolo-coco/yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names that we need from YOLO
print("\n------------------------------------------------------------------------------------------------------------")
print("[INFO] Loading Network Weights & Configuration from disk...")
print("-----------------------------------------------------------------------------------------------------------")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#
# cap1 = cv2.VideoCapture('/home/denslin/ML/trafic lights/car detection 2/videos/cars3.mp4')
# cap2 = cv2.VideoCapture('/home/denslin/ML/trafic lights/car detection 2/videos/cars3.mp4')
# cap3 = cv2.VideoCapture('/home/denslin/ML/trafic lights/car detection 2/videos/cars3.mp4')
# cap4 = cv2.VideoCapture('/home/denslin/ML/trafic lights/car detection 2/videos/cars3.mp4')
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
# cap3 = cv2.VideoCapture(2)
# cap4 = cv2.VideoCapture(2)


(W, H) = (None, None)

print("\n-----------------------------------------------------------------------------------------------------------")
print("[INFO] DETECTING VEHICALS ")
print("-----------------------------------------------------------------------------------------------------------")

while True:
    ret, frame = cap2.read()
    ret1,frame1=cap2.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap2.read()
    width = int(cap1.get(3))
    height = int(cap1.get(4))

    image = np.zeros(frame.shape, np.uint8)

    smaller_frame1 = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    smaller_frame2 = cv2.resize(frame1,(0,0),fx=0.5,fy=0.5)
    smaller_frame3 = cv2.resize(frame2,(0,0),fx=0.5,fy=0.5)
    smaller_frame4 = cv2.resize(frame3,(0,0),fx=0.5,fy=0.5)

    image[:height//2,:width//2]=smaller_frame1
    image[height//2:,:width//2]=smaller_frame2
    image[:height//2,width//2:]=smaller_frame3
    image[height//2:,width//2:]=smaller_frame4

    if W is None or H is None:
        (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    freq = []
    var = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # freq = [boxes.count(i) for i in boxes]
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5, 0.5)
    try:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    except:
        cv2.imshow('frame', image)
    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()