import cv2
import mediapipe as mp
import numpy as np
import time, os

filepath = 'C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_video/sick.mp4'


def Video_capture_save(filepath):
    cap = cv2.VideoCapture(filepath)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/sick/frame%d.jpg" % count, frame)
            print('Saved frame%d.jpg' % count)
            count += 1
    cap.release()
    cv2.destroyAllWindows()

Video_capture_save(filepath)
