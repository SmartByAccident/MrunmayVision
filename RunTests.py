import random
import time
from timeit import timeit
import cv2
import math
import numpy as np

def isConvex(origin, point1, point2):  
    """
    This function returns 1 if the angle formed by the three points is convex, -1 if it is concave, and 0 if it is collinear.
    """
    origin = origin[0]
    point1 = point1[0]
    point2 = point2[0]
    oX = origin[0]
    oY = origin[1]
    p1X = point1[0]
    p1Y = point1[1]
    p2X = point2[0]
    p2Y = point2[1]
    val = (p1X - oX)*(p2Y - oY) - (p2X - oX)*(p1Y - oY)
 
    if val > 0 :
        return False 
    else:
        return True

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
def findObjects(img, contours, id):
    new_contours = []
    objects = []
    image_copy = img.copy()
    for contour in contours:
        
        contourArea = cv2.contourArea(contour)
        #cv2.circle(img, (int(contour[0][0][0]), int(contour[0][0][1])), 10, (0, 0, 255), 2)
        # print(contourArea)
        if contourArea >= 2000 and contourArea <= 6000:
            
            contour = cv2.approxPolyDP(contour, 9,closed = True)
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            new_contours.append(contour)
            angle = orientation(contour)
            objects.append(((cx*3/64,cy*3/64),angle))
        elif contourArea >= 6000 and contourArea <= 10000:
            contour = cv2.approxPolyDP(contour, 6,closed = True)
            print("contour:",contour)
            for i in range(len(contour)):
                print("i:",i)
                print(isConvex(contour[i], contour[i-1], contour[(i+1)%len(contour)]))
                if isConvex(contour[i], contour[i-1], contour[(i+1)%len(contour)]):
                    print("convex",i)
                    for j in range(i+1,len(contour)):
                        if isConvex(contour[j], contour[j-1], contour[(j+1)%len(contour)]):
                            print("convex2",j)
                            print(i,j)
                            contour1 = np.concatenate((contour[:i+1],contour[j:]))
                            print("contour1",contour1)
                            contour2 = contour[i:j+1]
                            print("contour2",contour2)
                            new_contours.append(contour1)
                            new_contours.append(contour2)
                            angle1= orientation(contour1)
                            angle2 = orientation(contour2)
                            M1 = cv2.moments(contour1)
                            M2 = cv2.moments(contour2)
                            try:
                                cx1 = int((M1['m10']/M1['m00']))
                                cy1 = int((M1['m01']/M1['m00']))
                                objects.append(((cx1*3/64,cy1*3/64),angle1))
                            except ZeroDivisionError:
                                continue
                            try:
                                cx2 = int((M2['m10']/M2['m00']))
                                cy2 = int((M2['m01']/M2['m00']))
                                objects.append(((cx2*3/64,cy2*3/64),angle2))
                            except ZeroDivisionError:

                                continue
                            break
                    break
            #if convex corners > 0:
                #convex check
            #else
                #ratio check
            
    cv2.drawContours(image=img, contours=new_contours, contourIdx=-1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.imshow("Contoured"+id,img)
    return (objects,img)
def processImage(img,id):
    original = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # yellowMask = cv2.inRange(original, (0,23,127),(160,255,255))
    # blueMask = cv2.inRange(original, (102,0,0),(255,131,65))
    # redMask = cv2.inRange(original, (0,0,110),(130,116,255))
    yellowMask = cv2.inRange(original, (80,100,115),(110,255,255))
    blueMask = cv2.inRange(original, (0,100,0),(90,255,255))
    redMask = cv2.inRange(original, (110,100,115),(255,255,255))
    yellowMask = cv2.bitwise_and(cv2.bitwise_not(redMask), yellowMask)
    contoursYellow, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursBlue, hierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursRed, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contoursBlue)
    img_contour = img
    neutralSamples,img_contour = findObjects(img_contour, contoursYellow,"neutralSamples"+str(id))
    blueSamples,img_contour = findObjects(img_contour, contoursBlue,"blueSamples"+str(id))
    redSamples,img_contour =  findObjects(img_contour, contoursRed,"redSamples"+str(id))
    cv2.imwrite("ImageResults\\Batch3\\test"+str(id)+".jpg",img_contour)
    return (redSamples, blueSamples,neutralSamples)

if __name__ == '__main__':
    for i in range(70,71):
        print("Image:",i)
        img = cv2.imread('Images\\Tests2\\test'+str(i).zfill(2)+'.jpg') 
        img = cv2.resize(img,(512,512)) 
        objects = processImage(img,i)
        print(objects)
        print(objects)
    # print(isConvex((0,0),(3,4),(5,6)))
    # print(isConvex((0,0),(5,6),(3,4)))
    # print(isConvex((0,0),(-1,0),(0,-7)))
    # print(isConvex((0,0),(0,-7),(-1,0)))
    # print(isConvex((0,0),(-1,0),(1,-1)))
cv2.waitKey(0)
