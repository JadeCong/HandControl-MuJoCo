#!/usr/bin/env python
# -*- coding:UTF-8 -*-



# the description of this script which is necessary
"""
This script is used for function test of realsense D435 for capturing the depth image of human hand.
"""



# import sys
# sys.path.append("D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/application/")
# from hand_detecter import HandDetector
# import cv2
# import numpy as np
# import os
#
#
# file_path = "D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/dataset/mpl/mpl_mujoco/depth_mpl0"
# file_name = os.listdir(file_path)
# print(file_name)
#
# while True:
#     depth_image = cv2.imread(os.path.join(file_path, file_name[0]), cv2.IMREAD_UNCHANGED)
#
#     # com = depth_image.max()
#     # print(com)
#     print(depth_image[64,64])
#     com = np.array([64,64, depth_image[70,70]])
#     print(com)
#
#     hd = HandDetector(depth_image, 1395.12, 1395.12)
#     depth_image_crop = hd.crop_area_3d(com)
#
#     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('RealSense', depth_image)
#     # cv2.imshow('RealSense', depth_image_crop)
#
#     key = cv2.waitKey(1)
#
#     if key & 0xFF == ord('q') or key == 27:
#         cv2.destroyAllWindows()
#         break






# import the dependencies modules and self-defined packages
import sys
sys.path.append("D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/application/")
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from hand_detector import HandDetector

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

print("let's get start...")

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))
        # images = np.hstack((color_image, depth_image))

        # normalized the depth_image frame and clip it for my useage
        # cv2.normalize(depth_image, depth_image, 0, 255, cv2.NORM_MINMAX, -1)

        # get the center of mass of the hand
        # print(depth_image[320, 240])
        # print(type(depth_image))
        # com = np.array([320, 240, depth_image[320, 240]])
        # if depth_image[320, 240] == 0:
        #     com[2] = 200

        hd = HandDetector(depth_image, 600, 600)
        com = hd.get_target_com(roi=(200, 400))
        depth_image = hd.crop_area_3d(com, size=(200, 200, 200), img_size=(100, 100))

        # normalize to [0, 255]
        depth_max = np.max(depth_image)
        depth_min = np.min(depth_image)
        depth_scale = (depth_max - depth_min) / 255.0
        depth_image = (depth_image - depth_min) / depth_scale
        #
        # depth_image = depth_image[int(320-50):int(320+50), int(240-50):int(240+50)]
        print("depth_image size:{}".format(np.shape(depth_image)))

        # com = depth_image.min()
        # print(com)
        # depth_image = hd.crop_area_3d(com=com)
        # print("cropped depth_image size:{}".format(np.shape(depth_image)))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_image)
        # cv2.imshow('RealSense', images)
        # cv2.imshow('RealSense', depth_colormap)
        key = cv2.waitKey(1)
        # time.sleep(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()

