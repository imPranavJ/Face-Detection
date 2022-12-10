# Face-Detection

# Pre-Requisites
`pip install opencv-python`

# Introduction
This program detects the faces using the `haarcascade_frontalface_default.xml` cascade.

 # Blurring Face
 Step 1:
 First of, we just crop the face part detected and name it as roi...
 Step2:
 Using  Open-CV we use the Gaussian blur to blur the cropped part roi.
 Step 4:
 And paste the blurred roi back to original image.
 
 # Blurring Background
 Here, again we crop the detected face and feed in our frame to the blur function and save the output as `blurred_image`.
 After, this we again paste the `detected_face` back to our blurred image.
 
