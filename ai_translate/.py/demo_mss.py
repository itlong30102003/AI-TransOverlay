import mss
import numpy as np
import cv2

sct = mss.mss()
monitor = sct.monitors[1]  # màn hình chính

while True:
    img = np.array(sct.grab(monitor))
    cv2.imshow("Screen", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
    if cv2.waitKey(1) == 27:  # ESC để thoát
        break
