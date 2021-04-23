import pyrealsense2 as rs
import numpy as np
import cv2

if __name__ == "__main__":
    saveflag = 1
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

            # print (color_frame.get_data())
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # save image
            if saveflag == 1:
                color_image = cv2.resize(color_image, (240, 320))
                cv2.imwrite("../handsample/1.jpg", color_image)
                saveflag = 0
            if saveflag == 0:
                print ("SAVE!")
                break
    finally:
        # Stop streaming
        pipeline.stop()