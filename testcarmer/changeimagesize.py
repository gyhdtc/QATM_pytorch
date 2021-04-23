import cv2
import numpy as np
import os

pic = cv2.imread("/home/gyh/QATM_pytorch/handtemplate/1.png", cv2.IMREAD_COLOR)
pic_n = cv2.resize(pic, (30, 40))
cv2.imwrite("/home/gyh/QATM_pytorch/handtemplate/1.png", pic_n)