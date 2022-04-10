import cv2
import numpy as np

faceClassif=cv2.CascadeClassifier('face_detector.xml')

#image =cv2.imread('personas.jpg')
cap=cv2.VideoCapture(0)

while True:
    ret,image=cap.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceClassif.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=[30,30],maxSize=[200,200])

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
    cv2.imshow('imagen',image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=faceClassif.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=[30,30],maxSize=[200,200])

cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
cv2.imshow('imagen',image)
cv2.waitKey(0)

#cap= cv2.VideoCapture()