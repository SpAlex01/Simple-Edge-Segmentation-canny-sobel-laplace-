import os
import cv2
from matplotlib import pyplot as plt
import sys

def laplacian(img):
    depth = cv2.CV_16S
    kernel_size = 3
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    src_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, depth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    cv2.imshow("laplacian", abs_dst)
    cv2.waitKey(0)

    return 0

def sobel(img):
    scale = 1
    delta = 0
    depth = cv2.CV_16S
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    grad_y = cv2.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow('sobel', grad)
    cv2.waitKey(0)

    return 0

def canny(img):
    blur = cv2.GaussianBlur(img,(3,3),0) #kernel de 5x5
    cannynoise = cv2.Canny(img,threshold1=205, threshold2=210)
    cannyfaranoise = cv2.Canny(blur,threshold1=205, threshold2=210)
    #vizualizare
    cv2.imshow('Canny aplicat cu blur', cannyfaranoise)
    cv2.imwrite('desktop.png', cannyfaranoise)
    cv2.imshow('Canny aplicat fara blur', cannynoise)
    cv2.waitKey(0)

    return 0

imgpath=sys.argv[1]
print(imgpath)
img =cv2.imread(imgpath)

if int(img.shape[1]<300) or int(img.shape[0]<300):
    img = cv2.resize(img,(int(img.shape[1]*2.), int(img.shape[0]*2.)))
#vizualizare
cv2.imshow('Frame1', img)

cv2.waitKey(0)


canny(img)
sobel(img)
laplacian(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#canny




