import cv2
import math
import numpy as np
def orientation(contour):
    # print(contour)
    try:
        max_len = 0
        max_edge = None
        for i in range(len(contour)-1):
            current = contour[i]
            next = contour[(i+1)] 
            nextX = next[0][0]
            nextY = next[0][1]
            currentX = current[0][0]
            currentY = current[0][1]
            dX = currentX - nextX
            dY = currentY - nextY
            length = ((dX)**2 + (dY)**2 )** (1/2)
            if length > max_len:
                max_len = length
                max_edge = (current,next)
        length = ((contour[-1][0][0] - contour[0][0][0])**2 + (contour[-1][0][1] - contour[0][0][1])**2)** (1/2)
        if length > max_len:
                max_len = length
                max_edge = (contour[-1],contour[0])
        dX = max_edge[1][0][0] - max_edge[0][0][0]
        dY = max_edge[1][0][1] - max_edge[0][0][1]
        theta = math.degrees(math.atan2(dY, dX))
        return theta+180
    except:
        return -1
# Read the original image
img = cv2.imread('Images\\test31.jpg') 
img = cv2.resize(img,(512,512))
original = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# cv2.blur(img, (15,15)) 
# Display original image
cv2.imshow('Original', cv2.cvtColor(original,cv2.COLOR_LAB2RGB))
# Create a black image, a window
win = cv2.namedWindow('image')
cv2.resizeWindow('image', 512, 512)
# win.resize ((512,512))

# create trackbars for color change
cv2.createTrackbar('L','image',0,255,lambda x:None)
cv2.createTrackbar('A','image',0,255,lambda x:None)
cv2.createTrackbar('B','image',0,255,lambda x:None)
cv2.createTrackbar('L2','image',0,255,lambda x:None)
cv2.createTrackbar('A2','image',0,255,lambda x:None)
cv2.createTrackbar('B2','image',0,255,lambda x:None)
cv2.createTrackbar('minPixels','image',0,5000,lambda x:None)
cv2.createTrackbar('epsilon','image',0,255,lambda x:None)
cv2.createTrackbar('theta','image',-360,360,lambda x:None)
# cv2.waitKey(0)
while(True):

    l = cv2.getTrackbarPos('L','image')
    a = cv2.getTrackbarPos('A','image')
    b = cv2.getTrackbarPos('B','image')
    l2 = cv2.getTrackbarPos('L2','image')
    a2 = cv2.getTrackbarPos('A2','image')
    b2 = cv2.getTrackbarPos('B2','image')
    minPixels = cv2.getTrackbarPos('minPixels','image')
    epsilon = cv2.getTrackbarPos('epsilon','image')
    theta = cv2.getTrackbarPos('theta','image')

    yellowMask = original
    # yellowMask = cv2.GaussianBlur(original, (29,29), 0)
    # yellowMask = cv2.inRange(yellowMask, (0, 0, 100), (115,100, 255))
    yellowMask = cv2.inRange(yellowMask, (l,a,b),(l2,a2,b2))

    # yellowMask = cv2.threshold(yellowMask,220,255,cv2.THRESH_BINARY)[1]

    cv2.imshow('Yellow Mask', yellowMask)
    contours, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    image_copy = img.copy()
    for contour in contours:
        if cv2.contourArea(contour) >= minPixels:
            
            # cv2.circle(image_copy, (int(contour[0][0][0]), int(contour[0][0][1])), 10, (0, 0, 255), 2)
            contour = cv2.approxPolyDP(contour, epsilon ,closed = True)
            # for i in contour:
                # cv2.circle(image_copy, (int(i[0][0]), int(i[0][1])), 10, (0, 0, 255), 2)
            # print(contour)
            new_contours.append(contour)
            angle = orientation(contour)
            if(angle != -1):
                cv2.line(image_copy,*contour[0],
                        (int(100*math.cos(math.radians(angle)))+contour[0][0][0],int(100*math.sin(math.radians(angle)
                                                                                                ))+contour[0][0][1]),(240,32,160),3)
                

    cv2.drawContours(image=image_copy, contours=new_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
    cv2.imshow('Contours', cv2.cvtColor(image_copy,cv2.COLOR_LAB2RGB))#image_copy)
                
    img = cv2.bitwise_and(original, original, mask=yellowMask)
    cv2.imshow('Masked',cv2.cvtColor(img, cv2.COLOR_LAB2RGB))
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
# Convert to graycsale
# cv2.destroyAllWindows()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (21,21), 0) 
img_blur2 = cv2.GaussianBlur(img_gray, (61,61), 0) 
cv2.imshow("Blurred", img_blur)
cv2.imshow("Blurred2", img_blur2)

cv2.waitKey(0)
        
#Yellow Mask = (0,23,127),(160,255,255)
#Red Mask = (0,0,110),(130,116,255)
#Blue Mask = (102,0,0),(255,131,65)
