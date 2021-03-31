#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np

print("Package is ready")
# # Import any image
img=cv2.imread("Desktop/picture.jpg")
cv2.imshow("JASIM",img)
cv2.waitKey(0)
# # convert BGR to GRAY
# 
img=cv2.imread("lena.jpg")
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Lena",img)
cv2.imshow("GRAY Lena",imggray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # Import a video
cap=cv2.VideoCapture("G:\KOLKATA\python.mp4")
while True:
    success,img=cap.read()
    cv2.imshow("Video",img)
    if cv2.waitkey(1) & 0xFF==ord('q'):
        break
# # Convert BGR to blur and canny
# Dialation and Erodation
img=cv2.imread("lena.jpg")
kernel=np.ones((5,5),np.uint8)
imgblur=cv2.GaussianBlur(img,(7,7),0)
imgcanny=cv2.Canny(img,200,250)
imgdialation=cv2.dilate(imgcanny,kernel,iterations=1)
imgeroded=cv2.erode(imgdialation,kernel,iterations=1)
#cv2.imshow("original image",img) 
#cv2.imshow("Blur image",imgblur)
cv2.imshow("Canny image",imgcanny)  
cv2.imshow("Dialation image",imgdialation)
cv2.imshow("Eroded image",imgeroded)

cv2.waitKey(0)
cv2.destroyAllWindows()
# # Resize and Cropping
img=cv2.imread("lambo.png")
print(img.shape)
imgResize=cv2.resize(img,(300,200))
print(imgResize.shape)
imgCrop=img[0:300,100:200]

cv2.imshow("Lambo",img)
#cv2.imshow(" Resize Lambo",imgResize)
cv2.imshow("Cropped image",imgCrop)

cv2.waitKey(0)
cv2.destroyAllWindows()
# # USE OWN IMAGE
img=cv2.imread("picture2.JPG")
print(img.shape)


imgResize=cv2.resize(img,(300,200))
print(imgResize.shape)

imBGR=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

imGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#imgCrop=img[100:150,200:250]

#cv2.imshow("Lambo",img)

plt.imshow(imgResize)
plt.show()

plt.imshow(imBGR)
plt.show()

plt.imshow(imGray)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
# # Shape and Text
#black image and white image
imgBlack=np.zeros((512,512))
imgWhite=np.ones((512,512))
img=np.zeros((512,512,3),np.uint8)
img2=np.zeros((512,512,3),np.uint8)
cv2.line(img,(0,0),(512,512),(0,0,255),2)
cv2.rectangle(img,(0,0),(220,320),(0,255,0),cv2.FILLED)
cv2.circle(img,(400,20),20,(255,0,255),2)
cv2.putText(img,"OpenCv",(40,400),cv2.FONT_HERSHEY_COMPLEX,3,(0,25,255),1)
#print(img)
#img[:]=255,0,0
#img2[200:300,100:200]=0,255,0
#cv2.imshow("Black image",imgBlack)
#cv2.imshow("White image",imgWhite)
#cv2.imshow("(255,0,0)",img)
#cv2.imshow("(0,255,0)",img2)
cv2.imshow("Line in image",img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Create any image based on array values
# 
img=np.random.rand(4,2)
imgShape=cv2.resize(img,(300,200))
cv2.imshow("Random image",img)
cv2.imshow("Random image resize",imgShape)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # Warp perspective of image
img=cv2.imread("card.jpeg")
width,height=250,350
pts1=np.float32([[64,280],[533,73],[443,206],[498,270]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgOutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("card",img)
cv2.imshow("Output",imgOutput)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# # Image stacking
img=cv2.imread("lena.jpg")
img=cv2.resize(img,(300,200))

#img2=cv2.imread("picture2.JPG")
#img2=cv2.resize(img,(300,200))


imgVer=np.vstack((img,img))
cv2.imshow("Vertical image",imgVer)

imgHor=np.hstack((img,img,img))
cv2.imshow("Horizontal Image",imgHor)

#imgAdd_to_Ver=np.vstack((imgVer,imgVer))
#cv2.imshow("Addition of two vertical image",imgAdd_to_Ver)

cv2.waitKey(0) 
cv2.destroyAllWindows()


# # Color Detection

# In[ ]:


import cv2
import numpy as np

def empty(a):
    pass
path = 'lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    #print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    cv2.imshow("Original",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()



# # Create particular color
import cv2
import numpy as np

def empty(a):
    pass
path = 'lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    #print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Original",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result",imgResult)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()




# # Contours or Shape detection
def getContours(img):
    
    contours,hierarachy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        
        
        
        area = cv2.contourArea(contours)
        print(area)
    
img=cv2.imread("shapes.png")

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)
getContours(imgCanny)

cv2.imshow("Original",img)
#cv2.imshow("Gray",imgGray)
#cv2.imshow("Blur",imgBlur)
cv2.imshow("Canny",imgCanny)


cv2.waitKey(0)    
cv2.destroyAllWindows()
import numpy as np
import cv2

img = cv2.imread('shapes.png')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.imshow("img", img)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
          cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
          cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 10:
        cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # Facemark of image
import cv2

faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread('lena.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("Result", img)
cv2.waitKey(0)import cv2
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("picture2.JPG")
imgResize=cv2.resize(img,(340,240))
imgGray=cv2.cvtColor(imgResize,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for x,y,w,h in faces:
    cv2.rectangle(imgResize,(x,y),(x+w,y+h),(255,255,0),2)
plt.imshow(imgResize)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# # openCV project 01:Color Detection

# In[ ]:





# In[ ]:




