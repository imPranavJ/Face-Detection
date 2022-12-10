import cv2

imgPath = 'test.jpg'
cascPath = 'Face_Recognition_mtcnn\face_recognition\haarcascade_frontalface_default.xml'
faceCasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(imgPath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

imgs = cv2.resize(img, (960, 1000))
cv2.imshow("Face Detection", imgs)
cv2.waitKey(0)
