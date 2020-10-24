import cv2
import numpy as np
import tt,gg
import glob
import os

Mimg = cv2.imread('E:/Leaf/New folder/leaf (13).JPG')
#Mimg=cv2.imread("F:/movie/Movie/VIDEO_TS/voc2010/VOCdevkit/VOC2010/JPEGImages/2010_005433.jpg")
#=cv2.imread("F:/movie/Movie/VIDEO_TS/voc2007/VOCdevkit/VOC2007/JPEGImages/000283.jpg")
#Mimg=cv2.imread('E:/Python2/OverLap/im/input (2).jpg')
#Mimg=cv2.edgePreservingFilter(Mimg, flags=cv2.NORMCONV_FILTER)
#tt.gammaCorrection(Mimg)
# gaussian_1 = cv2.GaussianBlur(Mimg, (3, 3), 10.0)
# # rawImage = cv2.addWeighted(rawImage, 1.5, gaussian_1, -0.5, 0, rawImage)
# Mimg = cv2.addWeighted(Mimg, 2.0, gaussian_1, -0.5, 0, Mimg)
# cv2.imshow('SSS',Mimg)

files = glob.glob('D:/PostThesis/Contvex/Test1/*')
for f in files:
    os.remove(f)

global Main

rows, cols,dim = Mimg.shape
if rows > 700 or cols > 500:
    Mimg = cv2.resize(Mimg, (700, 500))
Main=Mimg
Mimg=tt.cow(Mimg)

#gaussian_1 = cv2.GaussianBlur(Mimg, (3, 3), 10.0)
#Mimg = cv2.addWeighted(Mimg, 1.5, gaussian_1, -0.5, 0, Mimg)

gray1 = cv2.cvtColor(Mimg, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray1)
cv2.imwrite('directory/Gray Image.jpg', gray1)
#gray_blur = cv2.medianBlur(gray1, 3)
gray_blur = gg.process(gray1, gg.build_filters())
#gray_blur = cv2.GaussianBlur(gray1, (15, 15), 0)
cv2.imshow('GrayBlur Image', gray_blur)
cv2.imwrite('directory/GrayBlur Image.jpg', gray_blur)
thresh1 = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 1)
#cv2.imshow('Thresh Image', thresh1)
#cv2.imwrite('directory/Thresh Image.jpg', thresh1)
#kernel3 = np.ones((3, 3), np.uint8)
#closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel3, iterations=4)
#cv2.imwrite('directory/Closing Image.jpg', closing1)

#image1 = cv2.imread("directory/Thresh Image.jpg")
#gray = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)  # grayscale
_, thresh = cv2.threshold(gray_blur, 150, 255, cv2.THRESH_BINARY_INV)
# threshold
cv2.imshow('Thresh Image 2', thresh1)
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel1, iterations=1)  # dilate

cv2.imshow("d", dilated)
cv2.imwrite('directory/dilated.jpg',dilated)
#cv2.waitKey(0)

#img=dilated
img = cv2.imread("directory/dilated.jpg")
img4=np.ones(img.shape, dtype=np.uint8)*255

img6=Main



image, contours3, hier = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours3 = sorted(contours3, key = cv2.contourArea, reverse = True)#[:15] # get largest five contour area
cc = contours3[1][1]
idx = 0
idx1 = 0
for cnt3 in contours3:
    # calculate epsilon base on contour's perimeter
    # contour's perimeter is returned by cv2.arcLength
    epsilon3 = 0.01 * cv2.arcLength(cnt3, True)

    # get approx polygons
    approx3 = cv2.approxPolyDP(cnt3, epsilon3, True)

    # hull is convex shape as a polygon
    hull = cv2.convexHull(cnt3)
    if len(cnt3) > 15:
        cv2.drawContours(img4, [hull], -1, (0, 0, 255))
        x, y, w, h = cv2.boundingRect(hull)
        #if w > 10 and h > 10:
        #------#-------print("%d %d  %d %d" % (x, y, w,h))
        idx += 1
        new_img = Main[y:y + h, x:x + w]
        cv2.imwrite('Test1/' + str(idx) + '.jpg', new_img)
cv2.imshow('contours2', img4)
cv2.imshow('Main',Main)
cv2.imshow('dsdda',image)
cv2.waitKey(0)


cv2.destroyAllWindows()



cv2.waitKey(0)
