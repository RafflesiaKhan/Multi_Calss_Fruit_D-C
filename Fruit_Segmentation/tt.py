import numpy as np
#import cv2
#from matplotlib import pyplot as plt
import cv2
from numpy import int8, uint8
#from pylab import array, plot, show, axis, arange, figure, uint8

def cow(im):
    # Edge preserving filter with two different flags.
    imout9 = cv2.edgePreservingFilter(im, flags=cv2.RECURS_FILTER);
    cv2.imwrite("Cow/edge-preserving-recursive-filter.jpg", imout9);

    imout8 = cv2.edgePreservingFilter(im, flags=cv2.NORMCONV_FILTER);
    cv2.imwrite("Cow/edge-preserving-normalized-convolution-filter.jpg", imout8);

    # Detail enhance filter
    imout7 = cv2.detailEnhance(im);
    cv2.imwrite("Cow/detail-enhance.jpg", imout7);

    # Pencil sketch filter
    imout_gray, imout = cv2.pencilSketch(im, sigma_s=60, sigma_r=0.07, shade_factor=0.05);
    cv2.imwrite("Cow/pencil-sketch.jpg", imout_gray);
    cv2.imwrite("Cow/pencil-sketch-color.jpg", imout);

    # Stylization filter
    cv2.stylization(im, imout);
    cv2.imwrite("Cow/stylization.jpg", imout);

    imout1 = cv2.edgePreservingFilter(im, flags=2, sigma_s=60, sigma_r=0.4)
    cv2.imwrite("Cow/edge-preserving-recursive-filter12.jpg", imout1);

    imout2 = cv2.detailEnhance(im, sigma_s=10, sigma_r=0.15)
    cv2.imwrite("Cow/detail-enhance12.jpg", imout2);

    dst_gray, dst_color = cv2.pencilSketch(im, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    cv2.imwrite("Cow/pencil-sketch12.jpg", dst_gray);
    cv2.imwrite("Cow/pencil-sketch-color12.jpg", dst_color);

    dst = cv2.stylization(im, sigma_s=60, sigma_r=0.07) # good
    #dst = cv2.stylization(im, sigma_s=60, sigma_r=0.07)
    cv2.imwrite("Cow/stylization12.jpg", dst);

    result= dst #good for overlapped/fruits
    #result = imout7 #good for Birds
    #result = imout2
    return result

def threshold(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("Cow/THRESH_BINARY.jpg", thresh1);
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("Cow/THRESH_BINARY_INV.jpg", thresh2);
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    cv2.imwrite("Cow/THRESH_TRUNC.jpg", thresh3);
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    cv2.imwrite("Cow/THRESH_TOZERO.jpg", thresh4);
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    cv2.imwrite("Cow/THRESH_TOZERO_INV.jpg", thresh5);
    #ret, thresh6 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result=thresh5
    return result

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def gammaCorrection(image):

    original = image
    idx=0
    for gamma in np.arange(0.0, 3.5, 0.5):
        if gamma == 1:
            continue
        idx = idx + 1
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, gamma=gamma)
        print("g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #cv2.imshow("Images", np.hstack([original, adjusted]))
        cv2.imwrite('Gamma/' + 'gam' + str(idx) + '.jpg', adjusted)
        if idx== 2:
           send=adjusted
        cv2.waitKey(0)
    return send

def thinn(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def median(img):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imshow("MideanF",dst)
    return dst

def bright(img):
    image = img  # ,0 load as 1-channel 8bit grayscale
    maxIntensity = 255.0  # depends on dtype of image data
    x = np.arange(maxIntensity)
    phi = 0.5#1
    theta = 0.5#1
    newImage0 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 0.5
    newImage0 = np.array(newImage0, dtype=uint8)
    y = (maxIntensity / phi) * (x / (maxIntensity / theta)) ** 0.5
    newImage1 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2
    newImage1 = np.array(newImage1, dtype=uint8)
    cv2.imwrite('directory/Brught.jpg', newImage1)
    return newImage1