#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
import glob
import pickle

import cv2
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *


class MPLPairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, input_size, input_viewpoint, is_train=False, with_name=False):
        self.input_size = input_size
        self.input_viewpoint = input_viewpoint #we got 9 viewpoint for different data sample
        self.data_path = data_path
        self.is_train = is_train
        self.with_name = with_name

        #TODO: 导出文件序列名称
        # if is_train:
        #     self.label = np.load(os.path.join(self.data_path, '/mpl/mpl_mujoco/depth_mpl0', 'hand_calculated_joints.txt')) #for training
        # else:
        #     self.label = np.load(os.path.join(self.data_path, '/mpl/mpl_mujoco/depth_mpl0', 'hand_calculated_joints.txt')) #for evaluation test
        if is_train:
            self.label = np.loadtxt(os.path.join(self.data_path, 'mpl/mpl_mujoco/depth_mpl0', 'hand_calculated_joints.txt'), dtype=str) #for training
        else:
            self.label = np.loadtxt(os.path.join(self.data_path, 'mpl/mpl_mujoco/depth_mpl0', 'hand_calculated_joints.txt'), dtype=str) #for evaluation test

        #return the size of dataset
        self.length = len(self.label)

    def __getitem__(self, index):
        tag = ''.join(self.label[index]).split(':', 1)
        fname = tag[0]
        # get the 20 joints for mpl each pose data frame
        # target = tag[1:].astype(np.float32)[-20:] #we got 20 joints(each finger has 4 joints) for mpl hand in mujoco
        angle = eval(tag[1])
        target = np.array([angle["thumb"],angle["index"],angle["middle"],angle["ring"],angle["pinky"]]).reshape(20,).astype(np.float32)
        
        # human = cv2.imread(os.path.join(self.data_path, 'human', 'human_crop', fname), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        human = cv2.imread(os.path.join(self.data_path, 'human', 'human_crop', '{}_crop.png'.format(fname)), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # TODO:(Done by Jade)make sure if the viewpoint[][0] is what
        viewpoint = self.input_viewpoint[np.random.choice(len(self.input_viewpoint), 1)][0]
        # TODO:(Done by Jade)figure out the crop stands for what
        # mpl = cv2.imread(os.path.join(self.data_path, 'mpl/mpl_crop', 'depth_mpl{}'.format(viewpoint), fname), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        mpl = cv2.imread(os.path.join(self.data_path, 'mpl/mpl_crop', 'depth_mpl{}'.format(viewpoint), 'depth_{}_crop.png'.format(fname)), cv2.IMREAD_ANYDEPTH).astype(np.float32)

        assert(human.shape[0] == human.shape[1] == self.input_size), "Wrong size for human image!"
        assert(mpl.shape[0] == mpl.shape[1] == self.input_size), "Wrong size for MPL image!"

        if self.is_train:
            # Augmented(if train)
            # 1. random rotated
            angle = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((self.input_size/2.0, self.input_size/2.0), angle, 1) #rotate the pic for angle degree around center of pic
            human = cv2.warpAffine(human, M, (self.input_size, self.input_size)) #return the rotated image for improving stochastic
            # mpl = cv2.warpAffine(mpl, M, (self.input_size, self.input_size))

            # 2. jittering
            min_human = np.min(human[human != 255.])
            max_human = np.max(human[human != 255.])
            delta = np.random.rand()*(255. - max_human + min_human) - min_human
            human[human != 255.] += delta
            human = human.clip(max=255., min=0.)

            # min_mpl = np.min(mpl[mpl != 255.])
            # max_mpl = np.max(mpl[mpl != 255.])
            # delta = np.random.rand()*(255. - max_mpl + min_mpl) - min_mpl
            # mpl[mpl != 255.] += delta
            # mpl = mpl.clip(max=255., min=0.)

        # Normalized
        human = human / 255. * 2. - 1
        mpl = mpl / 255. * 2. - 1

        human = human[np.newaxis, ...]
        mpl = mpl[np.newaxis, ...]

        if self.with_name:
            return mpl, human, target, fname
        else:
            return mpl, human, target

    def __len__(self):
        return self.length #return the size of dataset


if __name__ == '__main__':
    pass

