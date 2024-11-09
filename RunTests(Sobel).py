import random
import time
from timeit import timeit
import cv2
import math
import numpy as np


def processImage(img):

    yellowMask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_RGB2HSV), (80,100,115),(110,255,255))
    img = cv2.bitwise_and(img,img,mask = yellowMask)
    img = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow("img",img)
    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)

    # Canny Edge Detection
    edges = cv2.Canny(image=img, threshold1=200, threshold2=400) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    # cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    cdstP = img
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    

if __name__ == '__main__':
    img = cv2.imread('Images\\Tests3\\small.png') 
    objects = processImage(img)
    # print(isConvex((0,0),(3,4),(5,6)))
    # print(isConvex((0,0),(5,6),(3,4)))
    # print(isConvex((0,0),(-1,0),(0,-7)))
    # print(isConvex((0,0),(0,-7),(-1,0)))
    # print(isConvex((0,0),(-1,0),(1,-1)))
cv2.waitKey(0)
