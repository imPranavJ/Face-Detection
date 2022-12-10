import cv2
import numpy as np

cap = cv2.VideoCapture(1)
faceCasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')

def blur_img(img, factor = 20):
   kW = int(img.shape[1] / factor)
   kH = int(img.shape[0] / factor)
    
   #ensure the shape of the kernel is odd
   if kW % 2 == 0: kW = kW - 1
   if kH % 2 == 0: kH = kH - 1
    
   blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
   return blurred_img

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_img = blur_img(frame, factor=10)
    b = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)

    faces = faceCasc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

        detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
        d = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        b[y:y+h, x:x+w] = d
        
        cv2.imshow('Face Detection', b[:,:,::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
