import numpy as np
import cv2

#Don't run this in Jupyter Notebook. You'll get mad, Trust Me.
# You need these files in your folder. I think they're
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# This reads in the image
img = cv2.imread('me.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Not sure what this line does
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#  This loop finds your eyes inside of your face.
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# This makes the image pop up in a window
cv2.imshow('img',img)
# waitKey tells you how long the window should be open. 0 is forever.
#If you want it open for a couple of the seconds try 1000, if you want it open any longer add a zero.
cv2.waitKey(0)

