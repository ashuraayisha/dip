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

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104284597-21c73480-5467-11eb-94b0-f86054acd60f.png)
![image](https://user-images.githubusercontent.com/72543662/104284772-6ce14780-5467-11eb-9e95-4ba7a5105a16.png)

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
image = cv2.imread('jpg.webp')
image = cv2.resize(image, (0, 0), None, .25, .25)
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv2.imshow('flower', numpy_horizontal_concat)
cv2.imwrite("pg1.png",image)
cv2.waitKey()

The function waitKey(n) is used to wait for n milliseconds.
destroyAllWindows() function to close all the windows.

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104418814-5bae3e80-552c-11eb-841b-645f991916fe.png)


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

## 3.Develop a program to find sum and mean of a set of images.Create n number of images and read the directory and perform operation.

Adding all images(matrix of images) and finding the mean.mean() does, is simply summing all elements and then dividing by the number of all elements summed.

import cv2
import os
path='E:\ip'
imgs=[]
dirs = os.listdir(path)
for file in dirs :
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
i=0
sum_image = []
for sum_image in imgs:
    read_image= imgs[i]
    sum_image = sum_image + read_image
    #cv2.imshow(dirs[i],imgs[i])
    i = i +1
cv2.imshow('sum',sum_image)
print(sum_image)
cv2.imshow('mean',sum_image/i)
mean=(sum_image/i)
print(mean)
cv2.waitKey()
cv2.destroyAllWindows() 

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104437076-5c5ec900-555c-11eb-9170-2487d8031fb4.png)
![image](https://user-images.githubusercontent.com/72543662/104437185-7ef0e200-555c-11eb-9afc-491043876131.png)

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
Color Spaces in image processing are the color modes on the basis image can be segmented in particular object and non-object in an image.

cv2.COLOR_BGR2GRAY: This code is used to convert BGR colored image to grayscale
cv2.COLOR_BGR2HSV : This code is used to change the BGR color space to HSV color space.
cv2.COLOR_BGR2RGB : This code is used to change the BGR color space to RGB color space.
cv2. cv2.COLOR_BGR2LAB: This code is used to change the BGR color space to LAB color space.

RGB color space:
Linear RGB values are raw data obtained from a camera sensor. The value of R, G, and B are directly proportional to the amount of light that illuminates the sensor. Preprocessing of raw image data, such as white balance, color balance, and chromatic aberration compensation, are performed on linear RGB values.

LAB color space :
The L*a*b* color space provides a more perceptually uniform color space than the XYZ model. Colors in the L*a*b* color space can exist outside the RGB gamut (the valid set of RGB colors). For example, when you convert the L*a*b* value [100, 100, 100] to the RGB color space, the returned value is [1.7682, 0.5746, 0.1940], which is not a valid RGB color. For more information
L – Represents Lightness.
A – Color component ranging from Green to Magenta.
B – Color component ranging from Blue to Yellow.

HSV color space :
The HSV (Hue, Saturation, Value) color space corresponds better to how people experience color than the RGB color space does. For example, this color space is often used by people who are selecting colors, such as paint or ink color, from a color wheel or palette.
H : Hue represents dominant wavelength.
S : Saturation represents shades of color.
V : Value represents Intensity.

import cv2
image=cv2.imread("img20.jpg")
cv2.imshow("before",image)
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

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104419688-aa100d00-552d-11eb-8f20-ceb984af9202.png)
![image](https://user-images.githubusercontent.com/72543662/104419781-ce6be980-552d-11eb-9f56-82b42c83ee94.png)
![image](https://user-images.githubusercontent.com/72543662/104419879-f65b4d00-552d-11eb-88df-215d7d3781c9.png)
![image](https://user-images.githubusercontent.com/72543662/104419907-ffe4b500-552d-11eb-8fab-5cd43d8c32e5.png)
![image](https://user-images.githubusercontent.com/72543662/104419967-11c65800-552e-11eb-8e52-346684bc0a42.png)

## 6.Develop a program to create an image from 2D array.
numpy.zeros() or np.zeros Python function is used to create a matrix full of zeroes. numpy.zeros() in Python can be used when you initialize the weights.
Image.fromarray converts this array into an image of height h and width w.

import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([200, 200, 3], dtype=np.uint8)
array[:,:100] = [100, 118, 0] 
array[:,100:] = [100,100, 245]  
img = Image.fromarray(array)
img.save('images.jpg')
img.show()
c.waitKey(0)

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104421431-2dcaf900-5530-11eb-9516-f5d512ac2558.png)

## 7.Develop a program to find the sum of neighbour of each element in the matrix.
Here for each elements of the matrix.we are going to find then corresponding sum of matrix.
numpy.zeros() or np.zeros Python function is used to create a matrix full of zeroes. numpy.zeros() in Python can be used when you initialize the weights.

import numpy as np
M = [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("question:\n", M)
print ("answer:\n", N)

**OUTPUT:**
question:
 [[1 1 1]
 [1 1 1]
 [1 1 1]]
answer:
 [[3. 5. 3.]
 [5. 8. 5.]
 [3. 5. 3.]]

## 8.Develop a program to operator overloading.

#include<iostream>
using namespace std;
class matrix 
{
        int m, n, x[30][30]; 
public:
        matrix(int a, int b)
       { 
                m=a;
                n=b;
       }
        matrix(){}
        void get();
        void put();
        matrix operator +(matrix);
}; 

void matrix:: get()
{  
        cout<<"\n Enter values into the matrix";
               for(int i=0; i<m; i++)
                       for(int j=0; j<n;j++)
                       cin>>x[i][j];

}

void matrix:: put()
{  
        cout<<"\n the sum of the matrix is :\n";
               for(int i=0; i<m; i++)
               {
                       for(int j=0; j<n;j++)
                       cout<<x[i][j]<<"\t";
                       cout<<endl;
               }
} 

matrix matrix::operator +(matrix b)
{   
        matrix c(m,n);
        for(int i=0; i<m; i++)
                for(int j=0; j<n; j++)
                c.x[i][j]= x[i][j] + b.x[i][j];
        return c;
}

int main()
{    
        int m,n;
        cout<<"\n Enter the size of the array";
        cin>>m>>n;
        matrix a(m,n) , b(m,n) , c;
        a.get();
        b.get();
        c= a+b;
        c.put();
        return 0;
}

**OUTPUT:**

![image](https://user-images.githubusercontent.com/72543662/104443303-46550680-5564-11eb-94de-62d800424569.png)



