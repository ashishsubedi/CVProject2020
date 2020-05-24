import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from imutils import contours

'''
Process:
1) Figure out the hand mask in the ROI
2) Find the fingertip of hand mask in the ROI
3) Track the tip and start drawing on the screen as well
4) Find out the tracked digit and save it
5) Repeat from 2-5 till end (Either by button press or some special gesture)
6) Display the result
'''

cap = cv2.VideoCapture(0)
sampling = False
sampled = False
hist = None


def sample(roi):
    '''Sample the roi and figure out the hand mask'''
    '''
    Process:
    1) From the roi, find out hand mask either by color sampling or by calculating histogram
    '''
    global hist
    h, w = roi.shape[:2]
    # Cvt to hsv
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Sample rectangles to gather histogram
    sample1 = hsv[h//3:h//3+50, 2*w//3:2*w//3+30]
    sample2 = hsv[2*h//3:2*h//3+50, 2*w//3:2*w//3+30]
    sample3 = hsv[2*h//3:2*h//3+50, w//2:w//2+30]

    # Hist is none when no sample is made and calculate histogram in H and S channel. After sample is taken, the hist variable has sample saved
    if(hist is None):
        hist = cv2.calcHist([sample1, sample2, sample3], [0, 1],
                            None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def calcBackProj(roi):
    '''
    Calculating backProjection of our hsv channel with sampled hist to find mask
    '''
    global hist
    h, w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # Performing convolution with circular dist to increase clarity
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    backProj = cv2.filter2D(backProj, -1, disc)
    ret, thresh = cv2.threshold(backProj, 30, 200, 0)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    return thresh, roi


def getTip(roi, mask):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if(len(cnts) > 0):
        c = max(cnts, key=cv2.contourArea)
        # randomColor = np.random.randint(0, 255, (3,))
        # cv2.drawContours(roi, [c], -1, tuple([int(x)
        #                                       for x in randomColor]), 2)

        M = cv2.moments(c)
        center = (int(M['m10']/M['m00']), int(M['m01'] / M['m00']))
        hull = cv2.convexHull(c)
        maxDistance = 0
        for point in c:
            dist = np.linalg.norm(center-point[0])
            if(dist > maxDistance):
                maxDistance = dist
                fingerPoint = tuple(point[0])
        # cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)
        cv2.circle(roi, fingerPoint, 5, (0, 0, 255), -1)

    return roi, fingerPoint


op = None
mask = None
roi = None
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = 5*(H//8), 5*(W//8)
    roi = frame[H//8:H//8+h, W//6:W//6+w]
    cv2.rectangle(frame, (W//8, H//8), (W//8+w, H//8+h), (0, 255, 0), 2)
    if(sampling):
        cv2.putText(frame, "Sampling Done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        hist = sample(roi)
        sampling = False
        sampled = True
    elif(sampled):
        mask, roi = calcBackProj(roi)
        roi, fingerPoint = getTip(roi, mask)

    elif(not sampling and not sampled):

        cv2.putText(frame, "Put your hand in green region and Press s to start sampling",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        cv2.rectangle(roi, (2*w//3, h//3),
                      (2*w//3+30, h//3+50), (0, 255, 0), 2)
        cv2.rectangle(roi, (2*w//3, 2 * h//3),
                      (2*w//3+30, 2 * h//3+50), (0, 255, 0), 2)
        cv2.rectangle(roi, (w//2, 2 * h//3),
                      (w//2+30, 2 * h//3+50), (0, 255, 0), 2)

    total = time.time()-start
    fps = 1/(total+0.00000000001)
    cv2.putText(frame, f'FPS: {round(fps,2)}',
                (frame.shape[1]-100, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    frame = cv2.resize(frame, (W, H))

    cv2.imshow('frame', frame)
    if(mask is not None):
        mask = cv2.resize(mask, (W, H))
        cv2.imshow('mask', mask)

    pressed_key = cv2.waitKey(1) & 0xFF
    if(pressed_key == ord('s')):
        sampling = True
        sampled = False
    elif pressed_key == 27:
        break

cv2.destroyAllWindows()
cap.release()
