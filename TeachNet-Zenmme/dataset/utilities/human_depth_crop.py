#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
import os.path
from PIL import Image
import numpy as np


file_path = "../human/human_msra15/"
file_name = os.listdir(file_path)

for fn in file_name:
    if fn[-3:] == 'png':
        image = Image.open(file_path + str(fn))
        width,hight = image.size
        print("width:{}, hight:{}".format(width, hight))
        print(np.array(image).shape)

        # make sure the crop size
        img = np.array(image)
        top = 0
        bottom = 0
        left = 0
        right = 0

        for m in range(0,128,1):
            if (img[m,:].max() - img[m,:].min())>200:
                top = m
                print(top)
                break

        for n in range(127,-1,-1):
            if (img[n,:].max() - img[n,:].min())>200:
                bottom = n
                print(bottom)
                break

        for i in range(0,128,1):
            if (img[:,i].max() - img[:,i].min())>200:
                left = i
                print(left)
                break

        for j in range(127,-1,-1):
            if (img[:,j].max() - img[:,j].min())>200:
                right = j
                print(right)
                break

        center_y = (top + bottom)//2
        center_x = (left + right)//2

        top_crop_temp = center_y - 50
        bottom_crop_temp = center_y + 50
        left_crop_temp = center_x - 50
        right_crop_temp = center_x + 50

        if top_crop_temp <=0:
            top_crop = 0
            bottom_crop = 100
        else:
            top_crop = top_crop_temp
            bottom_crop = bottom_crop_temp

        if left_crop_temp <=0:
            left_crop = 0
            right_crop = 100
        else:
            left_crop = left_crop_temp
            right_crop = right_crop_temp

        if bottom_crop_temp >=128:
            bottom_crop = 127
            top_crop = 27
        else:
            top_crop = top_crop_temp
            bottom_crop = bottom_crop_temp

        if right_crop_temp >=128:
            right_crop =127
            left_crop = 27
        else:
            left_crop = left_crop_temp
            right_crop = right_crop_temp

        print("top:{},bottom:{},left:{},right:{}".format(top_crop,bottom_crop,left_crop,right_crop))
        img_crop = image.crop((left_crop,top_crop,right_crop,bottom_crop))
        width_crop,hight_crop = img_crop.size
        print("width_crop:{}, hight_crop:{}".format(width_crop, hight_crop))

        img_resize = img_crop.resize((100,100))
        img_resize.save('../human/human_crop/{}_{}.png'.format(fn[0:-4],'crop'))

