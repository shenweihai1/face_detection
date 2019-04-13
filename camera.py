import numpy as np
import cv2
import datetime 
import time
import os

cap = cv2.VideoCapture(0)
count = 1

try:
    os.stat("tmp2")
except:
    os.mkdir("tmp2")

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count += 1
    cv2.imwrite("tmp2/frame%d.jpg" % count, frame)
    time.sleep(1)
    pre_gray = cv2.imread("tmp2/frame%d.jpg" % count, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', pre_gray)
    print(datetime.datetime.now())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
