#!/usr/bin/env python
# -*- coding:UTF-8 -*-



# the description of this script which is necessary
"""
This script is for detect the hand form a depth image and crop hand form the
depth image as specified size.
"""



# import the dependencies modules and self-defined packages
import sys
sys.path.append("D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/")
import os
import numpy as np
import cv2



# define the hand detector class
class HandDetector(object):
    """ Detects the hand in a depth image.

    Assumes that the hand is the closest object to the camera. Uses the Center
    of Mass to crop the hand.

    Attributes:
        img: Depth image to crop from(in the pixel container is the real distance along the z axis in the real world)
        max_depth: Maximum depth value(unit:mm)
        min_depth: Minimum depth value(unit:mm)
        fx: Camera focal length in x dimension(the physic parameter of image device which you are using now)
        fy: Camera focal length in y dimension(the physic parameter of image device which you are using now)
        importer: Data importer object
    """

    def __init__(self, img, fx, fy, importer=None):
        """Constructor

        Initializes a new HandDetector object.

        Args:
            img: Input depth image
            fx: Camera focal length in x dimension
            fy: Camera focal length in y dimension
            importer: Data importer object
        """

        #TODO: make sure that the hand is in the range of given min and max
        self.img = img
        self.max_depth = min(2000, img.max()) #get at most 1500mm
        self.min_depth = max(10, img.min()) #get at least 100mm

        # Values out of range are 0
        self.img[self.img > self.max_depth] = 0.0 #only get the objects in the work range(between the 100mm and 1500mm)
        self.img[self.img < self.min_depth] = 0.0

        self.fx = fx
        self.fy = fy
        self.importer = importer


    def get_closet_value(self, img, ref):
        '''get the closet value to the given specified value in a ndarray'''
        size = np.shape(img)
        temp = 100.0
        for idx_i in range(size[0]):
            for idx_j in range(size[1]):
                target = img[idx_i, idx_j]
                if target == 0.0:
                    continue
                elif np.abs(target - ref) < temp:
                    temp = np.abs(target - ref)
                    index = (idx_i, idx_j)

        return index, img[index[0], index[1]]


    def get_closet_value_fast(self, img, ref):
        '''get the closet value to the given specified value in a ndarray'''
        temp = np.abs(img-ref)
        index = np.where(temp == np.min(temp))
        value = img[index[0][0], index[1][0]]

        return (index[0][0], index[1][0]), value


    def get_target_com(self, roi=(200, 300)):
        """Get the com of the target roi(region of interest)"""
        # only get the roi from the image
        img = self.img.copy()
        img[img > roi[1]] = 0.0
        img[img < roi[0]] = 0.0
        print(type(img))
        print(img.shape)

        # get the the average value of the roi
        average_img = np.mean(img[img != 0.0])
        if average_img == np.NaN:
            average_img = 300.0
        print("average:{}".format(average_img))

        # find the array index that the value of the index is closet to the specified value
        index, value = self.get_closet_value_fast(img, average_img)
        print("index {}: value {}".format(index, value))
        # index = np.where(np.floor(img) == int(average_img))
        # print("index {}: value {}".format(index, img[index[0], index[1]]))

        com = (index[0], index[1], value)

        return com


    def get_bounds_from_com(self, com, size):
        """Calculates the boundaries of the crop given the crop size and center
        of mass.

        The values are projected from image space to world space before adding
        the bounding box. The value is then projected back to image space.

        Args:
            com: Center of Mass in mm(the object in the image)
            size: 3D bounding box size in mm(generally the size=[250,250,250] is the real-world space)

        Returns:
            xstart: start of boundary in x dimension
            xend: end of boundary in x dimension
            ystart: start of bounadry in y dimension
            yend: end of bounadry in y dimension
            zstart: start of boundary in z dimension
            zend: end of bounadry in z dimension
        """

        zstart = com[2] - size[2] / 2.0
        zend = com[2] + size[2] / 2.0

        xstart = int(np.floor((com[0] * com[2] / self.fx - size[0] / 2.0) / com[2] * self.fx))
        xend = int(np.floor((com[0] * com[2] / self.fx + size[0] / 2.0) / com[2] * self.fx))
        ystart = int(np.floor((com[1] * com[2] / self.fy - size[1] / 2.0) / com[2] * self.fy))
        yend = int(np.floor((com[1] * com[2] / self.fy + size[1] / 2.0) / com[2] * self.fy))

        # crop_bounds = np.array([xstart, xend, ystart, yend, zstart, zend], dtype=int)
        crop_bounds = [xstart, xend, ystart, yend, zstart, zend]

        return crop_bounds


    def crop_img(self, img, crop_bounds, thresh_z=True):
        """Crops the given image using the specified boundaries.

        Args:
            img: Input image
            xstart: Starting value for x-axis bound
            xend: Ending value for x-axis bound
            ystart: Starting value for y-axis bound
            yend: Ending value for y-axis bound
            zstart: Starting value for z-axis bound
            zend: Ending value for z-axis bound
            thresh_z: Boolean to determine if z-values should be thresholded

        Returns:
            A cropped image.
        """

        xstart = crop_bounds[0]
        xend = crop_bounds[1]
        ystart = crop_bounds[2]
        yend = crop_bounds[3]
        zstart = crop_bounds[4]
        zend = crop_bounds[5]

        if len(img.shape) == 2:
            cropped_img = img[max(ystart, 0):min(img.shape[0], yend), max(xstart, 0):min(img.shape[1], xend)].copy()
            # fill in pixels if crop is outside of image
            #TODO:make sure that the image shape direction to get the right crop
            cropped_img = np.pad(cropped_img, ((abs(ystart) - max(ystart, 0),
                                            abs(yend) - min(yend, img.shape[0])),
                                            (abs(xstart) - max(xstart, 0),
                                            abs(xend) - min(xend, img.shape[1]))),
                                            mode='constant', constant_values=0)
        else:
            raise NotImplementedError()

        #TODO:make sure that the z threshold right
        if thresh_z is True:
            near_values = np.bitwise_and(cropped_img < zstart, cropped_img != 0)
            far_values = np.bitwise_and(cropped_img > zend, cropped_img != 0)
            cropped_img[near_values] = zstart
            cropped_img[far_values] = 0.0

        return cropped_img


    def crop_area_3d(self, com=None, size=(250, 250, 250), img_size=(100, 100)):
        """Performs a 3D crop of the hand.

        Given an input image, a 3D crop centered on the Center of Mass is
        returned.

        Args:
            com: Center of Mass
            size: Size of crop in 3D
            img_size: Output size of cropped image

        Returns:
            A 2D numpy array containing the cropped hand.
        """

        if len(size) != 3:
            raise ValueError("size must be 3D.")

        if len(img_size) != 2:
            raise ValueError("img_size must be 2D")

        if com is None:
            raise ValueError("CoM must be provided.")

        crop_bounds = self.get_bounds_from_com(com, size)

        cropped_img = self.crop_img(self.img, crop_bounds)

        # resize to requested image size
        cropped_img = cv2.resize(cropped_img, img_size, interpolation=cv2.INTER_NEAREST)

        return cropped_img

