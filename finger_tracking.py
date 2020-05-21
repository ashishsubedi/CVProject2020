import cv2
import numpy as np
import time
X = 2
Y = 3
cap = cv2.VideoCapture(0)

sampling = False
sampled = False


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
            req_img_roi = cv2.medianBlur(req_img_roi, 3)

            h.append(req_img_roi[:, :, 0])
            s.append(req_img_roi[:, :, 1])
            v.append(req_img_roi[:, :, 2])

    h = np.array(h)
    s = np.array(s)
    v = np.array(v)
    h_low, h_max = h.min(), h.max()
    s_low, s_max = s.min(), s.max()
    v_low, v_max = v.min(), v.max()
    return np.array([h_low-offset, s_low-offset, v_low-offset]), np.array([h_max+offset, s_max+offset, v_max+offset])


def start_sampling(frame, x=2, y=2, size=15, offset=5):
    frame, cords = draw_rect(frame, x, y, size=size)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low, high = create_mask(hsv, cords, x, y, size=size, offset=offset)
    mask = cv2.inRange(hsv, low, high)
    return mask


op = None
while True:
    start = time.time()
    pressed_key = cv2.waitKey(1)
    if(pressed_key == ord('s')):
        sampling = True
        sampled = False
    if(pressed_key == ord('d')):
        sampling = False
        sampled = True

    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    if(sampling):
        mask = start_sampling(frame)
        op = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(frame, "Sampling Started, put your palm to cover all rectangles.Press d when done",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7, 7))

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
        cv2.imshow('mask', mask)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
