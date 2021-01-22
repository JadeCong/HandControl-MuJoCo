#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import sys
import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

from data_importer.util.handdetector import HandDetector
from data_importer.util.helpers import shuffle_many_inplace
from data_importer.util.handpose_evaluation import ICVLHandposeEvaluation, NYUHandposeEvaluation, MSRAHandposeEvaluation
from data_importer.data.importers import MSRA15Importer
from data_importer.data.dataset import MSRA15Dataset

from pathlib import Path


CURRENT_DIR = Path(__file__).parent


class MSRA15(object):
    def __init__(self, sequence=None):
        self.sequence = sequence
        rng = np.random.RandomState(23455) #随机数种子值
        print('creating data for human hand from MSRA15 sequences.....')
        
        self.hpe = MSRAHandposeEvaluation(np.zeros((3, 3)), np.zeros((3, 3)))
        aug_modes = ['com', 'rot', 'none']
        comref=None
        docom=False

        dataset_path = os.path.join(str(CURRENT_DIR.parent),"human/MSRA15")
        cache_path = os.path.join(str(CURRENT_DIR.parent),"human/MSRA15_CACHE")
        di = MSRA15Importer(dataset_path, cacheDir=cache_path,refineNet=comref)
        
        print("you got the sequnece:P{}".format(self.sequence))
        Seqs = di.loadSequence('P{}'.format(self.sequence), shuffle=True, rng=rng, docom=docom)
        seqs=[Seqs]
        self.data = seqs[0].data
        
    
    def show_depth_img(self, index=None):
        print("you now choosing the image of sequence:P{} index:{}".format(self.sequence,index))
        plt.imshow(self.data[index].dpt)
        
        
    def draw_point(self, step=20):
        bg_img = np.ones(300*300).reshape(300,300)
        annoscale = 1
        joint = self.data.gtorig
        ax = bg_img
        hpe = self.hpe

        for i in range(step):
            lc = tuple((hpe.jointConnectionColors[i]*255.).astype(int))
            cv2.line(ax, (int(np.rint(joint[hpe.jointConnections[i][0], 0])),
                          int(np.rint(joint[hpe.jointConnections[i][0], 1]))),
                     (int(np.rint(joint[hpe.jointConnections[i][1], 0])),
                      int(np.rint(joint[hpe.jointConnections[i][1], 1]))),
                     (102,0,0), thickness=3*annoscale, lineType=cv2.LINE_AA)
            plt.imshow(ax)
    
    
    def get_gtorig_shape(self, index=None):
        return self.data[index].gtcrop.shape

    