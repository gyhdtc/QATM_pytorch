from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse

import multiprocessing
import pyrealsense2 as rs
from ctypes import c_wchar_p
import numpy as np
import cv2
# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys

from utils import *

with open("qatm_pytorch.py") as f:
       p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code,  module.__dict__)

from mod import *
# -

# global
global image_index
imageflag = 0
qatmflag = 0
# global
# lock
import threading

# lock

def getimage(image_index, image_name, global_lock_1, global_lock_2) -> None:
    print ("begin get image......" + str(image_index))
    saveflag = 1
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:       
        while image_index.value < 10:

            global_lock_1.acquire()
            
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            image_index.value += 1

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # save image
            color_image = cv2.resize(color_image, (320, 240))
            cv2.imwrite("handsample/1.jpg", color_image)
            image_name.value = str(image_index.value)+".jpg"
            print ("SAVE! - " + image_name.value)

            global_lock_2.release()
            
    finally:
        # Stop streaming
        pipeline.stop()
    
def showimage():
        # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
def GYH(image_index, image_name, global_lock_1, global_lock_2):
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default='sample/1.jpg')
    parser.add_argument('-t', '--template_images_dir', default='template/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    template_dir = args.template_images_dir
    image_path = args.sample_image
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    while image_index.value < 10:
        global_lock_1.release()
        global_lock_2.acquire()
        for image_file in os.listdir(Path(image_path)):
            # 多个模板匹配，暂时不需要
            # dataset,每个元素有：一个模板，一个相同的目标图片，一个图片名
            # image
            # image_raw
            # image_name
            # template
            # template_name
            # template_h
            # template_w
            # thresh
            dataset = ImageDataset(Path(template_dir), image_path+"/"+image_file, thresh_csv='thresh_template.csv')

            # print("calculate score..." + image_file)
            # scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
            # print("nms..." + image_file)
            # boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            # _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name="result-"+str(image_index.value)+".png")
            # print("result-" + "result-"+str(image_index.value)+".png" + " was saved")
            
            # 模拟一张图片匹配
            w_array = 0
            h_array = 0
            thresh = 0.8
            score = run_one_sample(model, dataset[0]['template'], dataset[0]['image'], dataset[0]['image_name'])
            w_array = dataset[0]['template_w']
            h_array = dataset[0]['template_h']

            boxes = nms(score, w_array, h_array, thresh)

            _ = plot_result(dataset[0]['image_raw'], boxes, show=False, save_name="result-"+str(image_index.value)+".png", color=(0,255,0))

if __name__ == '__main__':
    global_lock_1 = multiprocessing.Lock()
    global_lock_2 = multiprocessing.Lock()
    
    image_index = multiprocessing.Value('d', 0)
    image_name  = multiprocessing.Value(c_wchar_p, '1')

    p1 = multiprocessing.Process(target=getimage, args=[image_index, image_name, global_lock_1, global_lock_2])
    p2 = multiprocessing.Process(target=GYH, args=[image_index, image_name, global_lock_1, global_lock_2])
    global_lock_1.acquire()
    global_lock_2.acquire()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # -------------------------------------------- #
    # p1 = multiprocessing.Process(target = getimage, args=[global_lock_1, global_lock_2])
    # p1.start()
    # p1.join()
    # p2 = multiprocessing.Process(target=GYH)
    # p2.start()
    # p2.join()
    # -------------------------------------------- #