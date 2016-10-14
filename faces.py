import numpy as np
import cv2
#utiliza el xml de haarcascade para caras 
cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

pic = cv2.imread('cara.jpg')
grises = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
caripelas = cascada_cara.detectMultiScale(
	grises
	,scaleFactor=1.05
        ,minNeighbors=5
        ,minSize=(30, 30)
        ,flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

"""
caripelas = cascada_cara.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
"""

for (x,y,w,h) in caripelas:
	cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
#	roi_gray = grises[y:y+h, x:x+w]
#	roi_color = pic[y:y+h, x:x+w]
	
cv2.imshow('foto con las caripelas detectadas',pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
