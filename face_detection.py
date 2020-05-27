import cv2
import numpy as np

#Use cv2. VideoCapture( ) to get a video capture object for the camera. Set up an infinite while loop
# and use the read() method to read the frames using the above created object. Use cv2.
cap = cv2.VideoCapture(0)

#We will learn how the Haar cascade object detection works.
# We will see the basics of face detection and eye detection using the Haar Feature-based Cascade Classifiers
# We will use the cv::CascadeClassifier class to detect objects in a video stream. Particularly, we will use the functions:
# cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP classifier
# cv::CascadeClassifier::detectMultiScale to perform the detection.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ##this read two values one boolean and one frame
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        contnue
                                                   #scaling factor #NOofneigh
    faces = face_cascade.detectMultiScale(frame,1.3,5)



    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,10,0),4)
    cv2.imshow("Video Frame", frame)
    cv2.imshow("Video gray Frame", gray_frame)
    #wait fot user input -q , then you will stop the loop
    #here 1 in wait key means program will wait for 1 ms in
    #ord returns the ascii vale
    #cv2.waitkey returns a 32 bit integer 0xff is a no which have 8 1s so the and of a 32 bit no with 81s give effectively
    ##the last 8 bit numbers  so we are converting a 32 bit no with 8 bit no and then comparing with the ascii value
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
