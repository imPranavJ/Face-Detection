import cv2
import numpy as np

cap = cv2.VideoCapture(1)
faceCasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCasc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        center_coordinates = x + w // 2, y + h // 2
        radius = w // 2
        cv2.circle(frame, center_coordinates, radius, (255, 255, 255), 2)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)
        
        roi = frame[y:y + h, x:x + w]
        roi = cv2.GaussianBlur(roi, (79, 79), 0)
        frame[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
        
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
