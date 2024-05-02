import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from keras.models import load_model
import tensorflow

import numpy as np
import math
import time

count=0
folder="Images/peace"
classify= Classifier("keras_model.h5","labels.txt")
labels=["A","B","C","Peace"]
#model = load_model("keras_model.h5", compile=False)




#model=tensorflow.keras.models.load_model("Model/Keras_model.h5")

cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
while True:
    success, img= cap.read()
    imageOutput= img.copy()
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

            prediction, index=classify.getPrediction(imgWhite,draw=False)
            

        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imgcrop, (300, hCal))
            imgResizeShape = imageResize.shape
            gapH = math.ceil((300 - hCal) / 2)
            imgWhite[gapH:gapH + hCal,:] = imageResize
            prediction, index=classify.getPrediction(imgWhite,draw=False)
        
        cv2.putText(imageOutput,labels[index],(x+10,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.rectangle(imageOutput,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)


        cv2.imshow("Image Cropped", imgcrop)
        cv2.imshow("Image white", imgWhite)


    cv2.imshow("Image",imageOutput)
    cv2.waitKey(1)


