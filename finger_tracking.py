import cv2
import numpy as np
import time
from imutils import contours
X = 2
Y = 2
cap = cv2.VideoCapture(0)

sampling = False
sampled = False
tracking = False
tracker = cv2.TrackerCSRT_create()
handHist = None


def draw_rect(frame, x=2, y=2, size=15):
    h, w, _ = frame.shape
    tl_x = int(h/3+10)
    tl_y = int(w/3)
    coords = []
    tempX = tl_x
    tempY = tl_y
    for i in range(y):
        for j in range(x):
            br_x = int(tempX+size)
            br_y = int(tempY+size)
            cv2.rectangle(frame, (tempX, tempY),
                          (br_x, br_y),
                          (0, 255, 0), 1)
            coords.append((tempX, tempY))
            tempX += int(tl_x/3)
        tempX = tl_x
        tempY += int(tl_y/3)
    coords = np.array(coords, np.uint16)
    return frame, coords


def create_mask(frame, cords, x=2, y=2, size=15, offset=5):
    averages = np.zeros((size*x, size*y, 3), dtype=np.uint8)
    img = frame.copy()

    h = []
    s = []
    v = []
    for i in range(y):
        for j in range(x):

            req_x = [cords[i*x+j][0], cords[i*x+j][0]+size]
            req_y = [cords[i*x+j][1], cords[i*x+j][1]+size]
            req_img_roi = img[req_y[0]:req_y[1], req_x[0]:req_x[1]]
            # req_img_roi = cv2.medianBlur(req_img_roi, 3)

            h.append(req_img_roi[:, :, 0])
            s.append(req_img_roi[:, :, 1])
            v.append(req_img_roi[:, :, 2])

    h = np.array(h)
    s = np.array(s)
    v = np.array(v)
    h_low, h_max = int(np.min(h)), int(h.max())
    s_low, s_max = int(np.min(s)), int(s.max())
    v_low, v_max = int(np.min(v)), int(v.max())

    return np.array([h_low, s_low, v_low]), np.array([h_max, s_max, v_max])


def start_sampling(frame, x=2, y=2, size=15, offset=5):
    frame, cords = draw_rect(frame, x, y, size=size)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low, high = create_mask(hsv, cords, x, y, size=size, offset=offset)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    op = cv2.bitwise_and(frame, frame, mask=mask)

    global handHist
    handHist = cv2.calcHist([op], [0, 1], None, [180, 256], [0, 180, 0, 256])
    handHist = cv2.normalize(handHist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return mask


def get_convex_hull(frame):
    img = frame.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0.2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (9, 9))

    cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
    maxDistance = 0
    fingerPoint = None
    if(len(cnts) > 0):
        for c in cnts:
            M = cv2.moments(c)
            center = (int(M['m10']/M['m00']), int(M['m01'] / M['m00']))
            hull = cv2.convexHull(c)

            for point in c:
                dist = np.linalg.norm(center-point[0])
                if(dist > maxDistance):
                    maxDistance = dist
                    fingerPoint = tuple(point[0])

    return hull, maxDistance, center, fingerPoint


def startTracking(frame, mask, rect):
    pass
    # global tracking, tracker
    # if not tracking:
    #     tracking = True
    #     tracker.init(frame, rect)
    # else:
    #     (success, box) = tracker.update(frame)

    #     if success:
    #         (x, y, w, h) = [int(v) for v in box]
    #         cv2.rectangle(frame, (x, y), (x + w, y + h),
    #                       (0, 255, 0), 2)
    #         return frame, (x, y, w, h)

    # return frame,  -1

    # Find same colored mask as sampled colors


op = None
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if(sampling):
        mask = start_sampling(frame, X, Y)
        tracking = False
        op = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(frame, "Sampling Started, put your palm to cover all rectangles.Press d when done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (13, 13))
        hull, distance, center, fingerPoint = get_convex_hull(mask)
        fingerRect = (fingerPoint[0]-5, fingerPoint[1]-5, 10, 10)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        cv2.line(frame, center, fingerPoint, (255, 0, 0), 3)
    elif(sampled):
        cv2.circle(frame, fingerPoint, 5, (0, 0, 255), -1)
        mask = start_sampling(frame, X, Y)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (13, 13))
        hull, distance, center, fingerPoint = get_convex_hull(mask)
        frame, value = startTracking(frame, mask, hull, center, fingerPoint)
        if(value != -1):
            (x, y, w, h) = value
            fingerPoint = (x, y)
            fingerRect = value

    elif(not sampling and not sampled):
        cv2.putText(frame, "Press s to start sampling",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    total = time.time()-start
    fps = 1/total
    cv2.putText(frame, f'FPS: {round(fps,2)}',
                (frame.shape[1]-100, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    frame = cv2.resize(frame, (W, H))
    if (op is not None):
        cv2.imshow('op', op)

    cv2.imshow('frame', frame)
    cv2.imshow('hsv', hsv)
    pressed_key = cv2.waitKey(1) & 0xFF
    if(pressed_key == ord('s')):
        sampling = True
        sampled = False
    elif(pressed_key == ord('d')):
        sampling = False
        sampled = True
    elif pressed_key == 27:
        break

cv2.destroyAllWindows()
cap.release()
