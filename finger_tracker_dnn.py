import cv2
import numpy as np
protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22
threshold = 0.4

tips = [4, 8, 12, 16, 20]

sampled = False
tracking = False

WIDTH = 300
HEIGHT = 300
# frame = cv2.imread("hand.jpg")

tracker = cv2.TrackerCSRT_create()


def sample(frame):
    global protoFile, weightsFile, tips
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (WIDTH, HEIGHT),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    points = {}

    for i in range(nPoints):
        # confidence map of corresponding body's part.

        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (W, H))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            # cv2.circle(frame, (int(point[0]), int(
            #     point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(i), (int(point[0]), int(
            #     point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            if(i in tips):
                points[i] = ((int(point[0]), int(point[1])))
                cv2.circle(frame, (int(point[0]), int(
                    point[1])), 4, (0, 255, 0), thickness=-1)
        # else:
        #     points.append(None)
    return points


cap = cv2.VideoCapture(0)
prev = (-1, -1)
ret, frame = cap.read()
canvas = np.zeros_like(frame)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    if(not sampled):
        points = sample(frame)
        sampled = True
        tracking = False
        if(len(points) > 0):
            point = points[8] or points[12]
            rect = (point[0], point[1], 20, 20)
    else:

        if(not tracking):
            tracking = True
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, rect)
        else:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.circle(frame, (x, y), 5,  (0, 0, 255), -1)
                if(prev[0] != -1):
                    # if(abs(x-prev[0]) and abs(y-prev[1]) > 5):
                    cv2.line(canvas, prev, (x, y), (255, 0, 0), 2)
                prev = (x, y)
            else:
                tracking = False
                sampled = False
        frame = cv2.add(frame, canvas)

    cv2.imshow('Output', frame)
    k = cv2.waitKey(1)
    if k == ord('c'):
        canvas = np.zeros_like(canvas)
    if k == 27:
        break
