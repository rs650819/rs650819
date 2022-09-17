import cv2

# capture frames from a video
cap = cv2.VideoCapture('video1.avi')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x,y,w,h) in cars:
	    cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('video2', frames)   
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
