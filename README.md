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

## Scaling

import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)

**OUTPUT:**
![image](https://user-images.githubusercontent.com/72543662/104287045-c008c980-546a-11eb-9aa5-b0de048a3caa.png)
![image](https://user-images.githubusercontent.com/72543662/104287191-f2b2c200-546a-11eb-923b-951a18f3ca6d.png)














