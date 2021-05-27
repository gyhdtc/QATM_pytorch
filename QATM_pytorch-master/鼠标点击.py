import cv2
import numpy as np
import os
#图片路径
imagepath= 'd:/QATM_pytorch/Video1-capture/'
filenames=os.listdir(imagepath)

a =[]
b = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)

if __name__ == '__main__':
    for filename in filenames:
        filepath = imagepath + filename
        img = cv2.imread(filepath)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        while len(a) != 10:
            a.append(0)
            b.append(0)
        print(filename,end=",")
        for i in range(len(a)):
            if i < len(a)-1:
                print(a[i],end=",")
                print(b[i],end=",")
            else:
                print(a[i],end=",")
                print(b[i])
        a.clear()
        b.clear()
        # break