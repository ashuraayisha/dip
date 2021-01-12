## 1.Develop a program to display grayscale image using read and write operation.

Grayscaling----is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.

The function imread() is used for reading an image
The function imwrite() is used to write an image in memory to disk and
The function imshow() in conjunction with namedWindow and waitKey is used for displaying an image in memory.

**PROGRAM1**

import cv2
import numpy as np
image = cv2.imread('pic1.jpeg',0) 
cv2.imshow('Original', image) 
cv2.waitKey() 

The cv2.imread() function returns a NumPy array representing the image.
The first argument is the file name. The image should be in the working directory (or) a full path of the image should be given.
The second argument is a flag which specifies the way image should be read. The different flags are described below:
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_GRAYSCALE : Loads image in gray-scale mode
cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

**PROGRAM2**

import cv2
import numpy as np

image = cv2.imread('cat.jpg')
image = cv2.resize(image, (0, 0), None, 1.00, 1.00)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('cat', numpy_horizontal_concat)
cv2.waitKey()

The function waitKey(n) is used to wait for n milliseconds.
destroyAllWindows() function to close all the windows.

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104284597-21c73480-5467-11eb-94b0-f86054acd60f.png)
![image](https://user-images.githubusercontent.com/72543662/104284772-6ce14780-5467-11eb-9e95-4ba7a5105a16.png)

## 2. Develop a program to perform linear transformation on image. (Scaling and rotation) 

**Scaling**: scaling transformation alters size of an object. In the scaling process, we either compress or expand the dimension of the object.

import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104287045-c008c980-546a-11eb-9aa5-b0de048a3caa.png)
![image](https://user-images.githubusercontent.com/72543662/104287191-f2b2c200-546a-11eb-923b-951a18f3ca6d.png)

**Rotation**:We have to rotate an object by a given angle about a given pivot point and print the new co-ordinates.

import cv2 
import numpy as np 
img = cv2.imread('img22.jfif') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('result',res) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104289132-9b622100-546d-11eb-8c09-1b7b01ff842e.png)
![image](https://user-images.githubusercontent.com/72543662/104289254-c51b4800-546d-11eb-93c2-e465c9418f9d.png)

## 4.Write a program to convert color image into gray scale and binary image

A binary image is the type of image where each pixel is black or white. You can also say the pixels as 0 or 1 value. Here 0 represents black and 1 represents a white pixel.
To convert an RGB image into a binary type image, we need OpenCV. So first of all, if we don’t have OpenCV installed, then we can install it via pip.
After that, read our image as grayscale. Grayscale is a simplified image and it makes the process simple. Below is the code to get grayscale data of the image.
After that, read our image as grayscale. Grayscale is a simplified image and it makes the process simple. Below is the code to get grayscale data of the image.Now show the image.

import cv2
image=cv2.imread("img19.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104290229-f6e0de80-546e-11eb-9b3e-28e51f98d0bd.png)
![image](https://user-images.githubusercontent.com/72543662/104290360-1bd55180-546f-11eb-80d3-c4a6a31607c1.png)

## 5.Write a program to convert color image into different color space.
import cv2
image=cv2.imread("img20.jpg")
cv2.imshow("old",image)
cv2.waitKey()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)
cv2.waitKey(0)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",lab)
cv2.waitKey(0)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",hls)
cv2.waitKey(0)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT:


