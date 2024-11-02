import cv2
 
# Read the original image
img = cv2.imread('test.jpg') 
img = cv2.resize(img,(512,512))
original = img
# cv2.blur(img, (15,15)) 
# Display original image
cv2.imshow('Original', original)
# Create a black image, a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,lambda x:None)
cv2.createTrackbar('G','image',0,255,lambda x:None)
cv2.createTrackbar('B','image',0,255,lambda x:None)
cv2.createTrackbar('R2','image',0,255,lambda x:None)
cv2.createTrackbar('G2','image',0,255,lambda x:None)
cv2.createTrackbar('B2','image',0,255,lambda x:None)

# cv2.waitKey(0)
while(True):

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    r2 = cv2.getTrackbarPos('R2','image')
    g2 = cv2.getTrackbarPos('G2','image')
    b2 = cv2.getTrackbarPos('B2','image')
    yellowMask = cv2.inRange(original, (0, 78, 175), (120, 255, 255))
    img = cv2.bitwise_and(original, original, mask=yellowMask)
    cv2.imshow('Masked',img)
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
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
