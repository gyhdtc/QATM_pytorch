# -*- coding:utf-8 -*- 
import cv2 
import numpy as np 
import time 
import os

if __name__ == '__main__': 
    filepath = 'c:/Users/gyh/Desktop/frame700.png'
    Img = cv2.imread(filepath) # 读入一幅图像 
    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV) # 把BGR图像转换为HSV格式 
    # print('HSV',HSV)
    print('shape',Img.shape)
    rgbmin = [255,255,255]
    rgbmax = [0,0,0]
    hsvsum = [0,0,0]
    for i in HSV:
        for j in i:
            print(j[0],j[1],j[2])
            for k in range(3):
                hsvsum[k] = hsvsum[k] + j[k]
                if j[k] > rgbmax[k]:
                    rgbmax[k] = j[k]
                if j[k] < rgbmin[k]:
                    rgbmin[k] = j[k]
    print(hsvsum[0])
    print(hsvsum[1])
    print(hsvsum[2])
    print(rgbmax)
    print(rgbmin)