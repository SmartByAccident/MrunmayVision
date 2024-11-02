import time
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
def findObjects(img, contours, color):
    ct = time.time()
    new_contours = []
    image_copy = img.copy()
    for contour in contours:
        contourArea  = cv2.contourArea(contour)
        if contourArea >= 2000 and contourArea <= 5500:
            contour = cv2.approxPolyDP(contour, 9,closed = True)
       
            new_contours.append(contour)
            angle = orientation(contour)
            if(angle != -1):
                cv2.line(image_copy,*contour[0],
                     (int(100*math.cos(math.radians(angle)))+contour[0][0][0],int(100*math.sin(math.radians(angle)
                                                                                               ))+contour[0][0][1]),(240,32,160),3)
        

    cv2.drawContours(image=image_copy, contours=new_contours, contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)
    return image_copy
    

# Read the original image
for i in range(1,43):
    img = cv2.imread('Images\\test'+str(i).zfill(2)+'.jpg') 
    img = cv2.resize(img,(512,512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    original = img #img
    # cv2.blur(img, (15,15)) 
    # Display original image
    cv2.imshow('Original', original)
    # Create a black image, a window
    # win = cv2.namedWindow('image')
    # cv2.resizeWindow('image', 512, 512)
    # win.resize ((512,512))

    # create trackbars for color change
    # cv2.createTrackbar('R','image',0,255,lambda x:None)
    # cv2.createTrackbar('G','image',0,255,lambda x:None)
    # cv2.createTrackbar('B','image',0,255,lambda x:None)
    # cv2.createTrackbar('R2','image',0,255,lambda x:None)
    # cv2.createTrackbar('G2','image',0,255,lambda x:None)
    # cv2.createTrackbar('B2','image',0,255,lambda x:None)
    # cv2.createTrackbar('minPixels','image',0,1000,lambda x:None)
    # cv2.createTrackbar('epsilon','image',0,255,lambda x:None)
    # cv2.createTrackbar('theta','image',-360,360,lambda x:None)
    # # cv2.waitKey(0)
    # # while(True):

    # r = cv2.getTrackbarPos('R','image')
    # g = cv2.getTrackbarPos('G','image')
    # b = cv2.getTrackbarPos('B','image')
    # r2 = cv2.getTrackbarPos('R2','image')
    # g2 = cv2.getTrackbarPos('G2','image')
    # b2 = cv2.getTrackbarPos('B2','image')
    # minPixels = cv2.getTrackbarPos('minPixels','image')
    # epsilon = cv2.getTrackbarPos('epsilon','image')
    # theta = cv2.getTrackbarPos('theta','image')

    # yellowMask = cv2.GaussianBlur(original, (29,29), 0)
    yellowMask = cv2.inRange(original, (80,100,115),(110,255,255))
    blueMask = cv2.inRange(original, (0,100,115),(80,255,255))
    redMask = cv2.inRange(original, (110,100,115),(255,255,255))
    # yellowMask = cv2.bitwise_and(cv2.bitwise_not(redMask), yellowMask)
    # yellowMask = cv2.inRange(yellowMask, (b, g, r), (b2,g2, r2))

    # yellowMask = cv2.threshold(yellowMask,220,255,cv2.THRESH_BINARY)[1]

    # cv2.imshow('Yellow Mask', yellowMask)
    # cv2.imshow('Blue Mask', blueMask)
    # cv2.imshow('Red Mask', redMask)
    contoursYellow, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursBlue, hierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursRed, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img
    img_contour = findObjects(img_contour, contoursYellow,(85,255,255))
    img_contour = findObjects(img_contour, contoursBlue,(190,255,255))
    img_contour =findObjects(img_contour, contoursRed,(255,255,255))
    print(type(img))
    cv2.imshow('Objects',img_contour) 
    cv2.imwrite('ImageResults\\testResults'+str(i).zfill(2)+'.jpg',cv2.cvtColor(img_contour,cv2.COLOR_HSV2BGR))
    

# img = cv2.bitwise_and(original, original, mask=yellowMask)
# cv2.imshow('Masked',img)
# if(cv2.waitKey(1) & 0xFF == ord('q')):
#     break
# Convert to graycsale
# cv2.destroyAllWindows()

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(img_gray, (21,21), 0) 
# img_blur2 = cv2.GaussianBlur(img_gray, (61,61), 0) 
# cv2.imshow("Blurred", img_blur)
# cv2.imshow("Blurred2", img_blur2)

# cv2.waitKey(0)
"""
R Y     FY
 0 0    0
 0 1    1
 1 0    0
1 1     0
        """
