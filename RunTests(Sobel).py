import random
import time
from timeit import timeit
import cv2
import math
import numpy as np


def processImage(img):

    yellowMask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_RGB2HSV), (80,100,115),(110,255,255))
    img = cv2.bitwise_and(img,img,mask = yellowMask)
    cv2.imshow("img",img)
    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)

    # Canny Edge Detection
    edges = cv2.Canny(image=img, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    

if __name__ == '__main__':
    img = cv2.imread('Images\\Tests3\\small.png') 
    objects = processImage(img)
    # print(isConvex((0,0),(3,4),(5,6)))
    # print(isConvex((0,0),(5,6),(3,4)))
    # print(isConvex((0,0),(-1,0),(0,-7)))
    # print(isConvex((0,0),(0,-7),(-1,0)))
    # print(isConvex((0,0),(-1,0),(1,-1)))
cv2.waitKey(0)
