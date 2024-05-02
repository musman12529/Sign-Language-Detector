import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

count=0
folder="Images/peace"

cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
while True:
    success, img= cap.read()
    hands, img= detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h= hand['bbox']
        imgWhite= np.ones((300,300,3),np.uint8)*255
        imgcrop=img[y-20:y+h+20, x-20:x+w+20]

        imgCropShape= imgcrop.shape


        aspectRatio = h/w

        if aspectRatio>1:
            k=300/h
            wCal= math.ceil(k*w)
            imageResize= cv2.resize(imgcrop, (wCal,300))
            imgResizeShape = imageResize.shape
            gapW=math.ceil((300-wCal)/2)
            imgWhite[:, gapW:gapW+wCal] = imageResize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imgcrop, (300, hCal))
            imgResizeShape = imageResize.shape
            gapH = math.ceil((300 - hCal) / 2)
            imgWhite[gapH:gapH + hCal,:] = imageResize


        cv2.imshow("Image Cropped", imgcrop)
        cv2.imshow("Image white", imgWhite)


    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key==ord("s"):
        count+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
