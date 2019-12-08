#For Numerical Operations
import numpy as np

#Computer Vision Library
import cv2

#Haarcascades classifier for face recongnition
#it is in xml format it will find the face x,y co-ordinates and height and width
face_classifier=cv2.CascadeClassifier('./Dataset/Haarcascades/haarcascade_frontalface_default.xml')

#read the image
mark_image=cv2.imread('./Dataset/mark.jpg')

#Display the image
cv2.imshow("mark",mark_image)
#eait for the Key to be pressed 
cv2.waitKey()
#If pressed tehn Destroy that image Windows
cv2.destroyAllWindows()


#convert the rgb color image to grayscale(black and white) so that processing is fast
mark_grey_image=cv2.cvtColor(mark_image,cv2.COLOR_BGR2GRAY)

#Display Gray color image
cv2.imshow("mark_GRAY",mark_grey_image)
cv2.waitKey()
cv2.destroyAllWindows()

#Detect the face
faces=face_classifier.detectMultiScale(mark_grey_image,1.3,5)

#Here the X and y axis of the face and then w and h are width 
#and height of the image 
print(faces)

#If the image has no faces then faces arry will be empty
if faces is ():
    print("NO Faces Found in the image")
else:
    print("Face Found")
    
for (x,y,w,h) in faces:
    cv2.rectangle(mark_image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow("mark",mark_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
    