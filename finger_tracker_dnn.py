import cv2
import numpy as np
protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22

threshold = 0.4
WIDTH = 300
HEIGHT = 300
# frame = cv2.imread("hand.jpg")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (WIDTH, HEIGHT),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (W, H))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frame, (int(point[0]), int(
                point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(point[0]), int(
                point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    cv2.imshow('Output-Keypoints', frame)
    if cv2.waitKey(1) == 27:
        break
