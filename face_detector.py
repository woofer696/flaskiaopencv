import cv2
import numpy as np

faceClassif=cv2.CascadeClassifier('face_detector.xml')

image =cv2.imread('personas.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=faceClassif.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=[30,30],maxSize=[200,200])

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
cv2.imshow('imagen',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cap= cv2.VideoCapture()