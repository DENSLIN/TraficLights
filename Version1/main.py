import numpy as np
# import argparse
# import imutils
import Algo.logicv1 as log
import time
import cv2

# import os
# from tqdm import tqdm
print("\n------------------------------------------------------------------------------------------------------------")
print("[INFO] Loading Network Weights & Configuration from disk...")
# load the COCO class labels our YOLO model was trained on
labelsPath = "E:/ml/trafic lights/car detection 2/yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "E:/ml/trafic lights/car detection 2/yolo-coco/yolov3.weights"
configPath = "E:/ml/trafic lights/car detection 2/yolo-coco/yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names that we need from YOLO

print("-----------------------------------------------------------------------------------------------------------")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# LOADED VIDEO CAPTURE
cap1 = cv2.VideoCapture('E:/ml/trafic lights/car detection 2/Version1/3X3 video/R1/R1.mp4')
cap2 = cv2.VideoCapture('E:/ml/trafic lights/car detection 2/Version1/3X3 video/R2/R2.mp4')
cap3 = cv2.VideoCapture('E:/ml/trafic lights/car detection 2/Version1/3X3 video/R3/R3.mp4')
cap4 = cv2.VideoCapture('E:/ml/trafic lights/car detection 2/Version1/3X3 video/R4/R4.mp4')

# LIVE VIDEO CAPTURE
# cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(2)
# cap3 = cv2.VideoCapture(2)
# cap4 = cv2.VideoCapture(2)
print("\n-----------------------------------------------------------------------------------------------------------")
print("[INFO] LOADING TRAFFIC NETWORK ")
smart_system = log.Network()
print("-----------------------------------------------------------------------------------------------------------")

(W, H) = (None, None)

print("\n-----------------------------------------------------------------------------------------------------------")
print("[INFO] DETECTING VEHICALS ")
print("-----------------------------------------------------------------------------------------------------------")

countPrev = [0, 0, 0, 0, 0, 0, 0, 0]

while True:
    ret, frame = cap1.read()
    ret1, frame1 = cap2.read()
    ret2, frame2 = cap3.read()
    ret3, frame3 = cap4.read()
    width = int(cap1.get(3))
    height = int(cap1.get(4))
    image = np.zeros(frame.shape, np.uint8)

    smaller_frame1 = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    smaller_frame2 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
    smaller_frame3 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    smaller_frame4 = cv2.resize(frame3, (0, 0), fx=0.5, fy=0.5)

    image[:height // 2, :width // 2] = smaller_frame1
    image[height // 2:, :width // 2] = smaller_frame2
    image[:height // 2, width // 2:] = smaller_frame3
    image[height // 2:, width // 2:] = smaller_frame4
    image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)

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
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    x1, y1 = 190, 90
    x2, y2 = 265, 90
    x3, y3 = 335, 90
    x4, y4 = 220, 395
    x5, y5 = 360, 395
    x6, y6 = 500, 395

    x7, y7 = 915, 90
    x8, y8 = 990, 90
    x9, y9 = 1060, 90
    x10, y10 = 940, 395
    x11, y11 = 1080, 395
    x12, y12 = 1220, 395

    x13, y13 = 190, 500
    x14, y14 = 265, 500
    x15, y15 = 335, 500
    x16, y16 = 220, 800
    x17, y17 = 360, 800
    x18, y18 = 500, 800

    x19, y19 = 915, 500
    x20, y20 = 990, 500
    x21, y21 = 1060, 500
    x22, y22 = 940, 800
    x23, y23 = 1080, 800
    x24, y24 = 1220, 800

    cv2.line(image, (x1, y1), (x4, y4), (0, 0, 0xFF), 2)
    cv2.line(image, (x2, y2), (x5, y5), (0, 0, 0xFF), 2)
    cv2.line(image, (x3, y3), (x6, y6), (0, 0, 0xFF), 2)
    cv2.line(image, (x1, y1), (x3, y3), (0, 0xFF, 0), 2)
    cv2.line(image, (x4, y4), (x6, y6), (0, 0xFF, 0), 2)

    cv2.line(image, (x7, y7), (x10, y10), (0, 0, 0xFF), 2)
    cv2.line(image, (x8, y8), (x11, y11), (0, 0, 0xFF), 2)
    cv2.line(image, (x9, y9), (x12, y12), (0, 0, 0xFF), 2)
    cv2.line(image, (x7, y7), (x9, y9), (0, 0xFF, 0), 2)
    cv2.line(image, (x10, y10), (x12, y12), (0, 0xFF, 0), 2)

    cv2.line(image, (x13, y13), (x16, y16), (0, 0, 0xFF), 2)
    cv2.line(image, (x14, y14), (x17, y17), (0, 0, 0xFF), 2)
    cv2.line(image, (x15, y15), (x18, y18), (0, 0, 0xFF), 2)
    cv2.line(image, (x13, y13), (x15, y15), (0, 0xFF, 0), 2)
    cv2.line(image, (x16, y16), (x18, y18), (0, 0xFF, 0), 2)

    cv2.line(image, (x19, y19), (x22, y22), (0, 0, 0xFF), 2)
    cv2.line(image, (x20, y20), (x23, y23), (0, 0, 0xFF), 2)
    cv2.line(image, (x21, y21), (x24, y24), (0, 0, 0xFF), 2)
    cv2.line(image, (x19, y19), (x21, y21), (0, 0xFF, 0), 2)
    cv2.line(image, (x22, y22), (x24, y24), (0, 0xFF, 0), 2)

    cv2.putText(image, "R1", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R2", (100, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R3", (820, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R4", (820, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0xFF), 2)

    cv2.putText(image, "R1S1", (400, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R1S2", (260, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R2S1", (400, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R2S2", (260, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R3S1", (1120, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R3S2", (1000, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R4S1", (1120, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)
    cv2.putText(image, "R4S2", (1000, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0xFF), 2)

    R1S1, R1S2, R2S1, R2S2, R3S1, R3S2, R4S1, R4S2 = 0, 0, 0, 0, 0, 0, 0, 0


    def checkx(y, x1, x2, y1, y2):
        return int((((x2 - x1) / (y2 - y1)) * (y - y1)) + x1)


    def checky(x, x1, x2, y1, y2):
        return int((((y2 - y1) / (x2 - x1)) * (x - x1)) + y1)


    def countInLane(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
        if ((checkx(y, x1, x2, y1, y2) < x and x < checkx(y, x3, x4, y3, y4)) and
                (checky(x, x1, x3, y1, y3) < y and y < checky(x, x2, x4, y2, y4))):
            if (classIDs[i] == 2 or classIDs[i] == 3 or classIDs[i] == 5 or classIDs[i] == 7):
                # print(x,y)
                return 1
        return 0


    # try:
    for i in idxs.flatten():
        (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        x, y = (x + (w / 2)), (y + (h / 2))
        R1S1 = R1S1 + countInLane(x, y, x2, y2, x5, y5, x3, y3, x6, y6)
        R1S2 = R1S2 + countInLane(x, y, x1, y1, x4, y4, x2, y2, x5, y5)

        R2S1 = R2S1 + countInLane(x, y, x14, y14, x17, y17, x15, y15, x18, y18)
        R2S2 = R2S2 + countInLane(x, y, x13, y13, x16, y16, x14, y14, x17, y17)

        R3S1 = R3S1 + countInLane(x, y, x8, y8, x11, y11, x9, y9, x12, y12)
        R3S2 = R3S2 + countInLane(x, y, x7, y7, x10, y10, x8, y8, x11, y11)

        R4S1 = R4S1 + countInLane(x, y, x20, y20, x23, y23, x21, y21, x24, y24)
        R4S2 = R4S2 + countInLane(x, y, x19, y19, x22, y22, x20, y20, x23, y23)

    cv2.putText(image, "R1S1: {}".format(R1S1), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R1S2: {}".format(R1S2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)

    cv2.putText(image, "R2S1: {}".format(R2S1), (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R2S2: {}".format(R2S2), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)

    cv2.putText(image, "R3S1: {}".format(R3S1), (730, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R3S2: {}".format(R3S2), (730, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)

    cv2.putText(image, "R4S1: {}".format(R4S1), (730, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)
    cv2.putText(image, "R4S2: {}".format(R4S2), (730, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0xFF), 2)

    count = ([R1S1, R1S2, R2S1, R2S2, R3S1, R3S2, R4S1, R4S2])

    if countPrev != count:
        print(smart_system.currentSignal.light)
        print(count)
        smart_system.solve(count)
    countPrev = count

    # except:
    #     cv2.imshow('frame', image)
    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()
