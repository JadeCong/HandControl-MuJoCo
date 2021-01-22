import numpy
import gc
import matplotlib
import sys

# sys.path.insert(0,'/data/dotfiles_xy/nvim/plugged/deep-prior-pp/src')

matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
from util.handdetector import HandDetector
import os
import numpy as np
from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset
from util.handpose_evaluation import MSRAHandposeEvaluation
from util.helpers import shuffle_many_inplace
from util.handpose_evaluation import ICVLHandposeEvaluation, NYUHandposeEvaluation, MSRAHandposeEvaluation

class Anyone(object):
    def __init__(self,ix=None):

        eval_prefix = 'MSRA15_COM_AUGMENT'
        if not os.path.exists('./eval/'+eval_prefix+'/'):
                        os.makedirs('./eval/'+eval_prefix+'/')

        rng = numpy.random.RandomState(23455)

        print("create data")
        aug_modes = ['com', 'rot', 'none']  # 'sc',

        comref=None
        docom=False
        di = MSRA15Importer('/odata/DataSets/cvpr15_MSRAHandGestureDB', refineNet=comref)
        Seq0 = di.loadSequence('P0', shuffle=True, rng=rng, docom=docom)
        # Seq1 = di.loadSequence('P1', shuffle=True, rng=rng, docom=docom)
        # Seq2 = di.loadSequence('P2', shuffle=True, rng=rng, docom=docom)
        # Seq3 = di.loadSequence('P3', shuffle=True, rng=rng, docom=docom)
        # Seq4 = di.loadSequence('P4', shuffle=True, rng=rng, docom=docom)
        # Seq5 = di.loadSequence('P5', shuffle=True, rng=rng, docom=docom)
        # Seq6 = di.loadSequence('P6', shuffle=True, rng=rng, docom=docom)
        # Seq7 = di.loadSequence('P7', shuffle=True, rng=rng, docom=docom)
        # Seq8 = di.loadSequence('P8', shuffle=True, rng=rng, docom=docom)
        # seqs = [Seq0, Seq1, Seq2, Seq3, Seq4, Seq5, Seq6, Seq7, Seq8]
        seqs = [Seq0]

        self.hpe = MSRAHandposeEvaluation(numpy.zeros((3, 3)), numpy.zeros((3, 3)))

        if ix:
            sample_ix= ix
        else:
            sample_ix= np.random.randint(1000)
            print(sample_ix)

        self.data=seqs[0].data[sample_ix]
        print("you now choosing seq 0 {}".format(sample_ix))
        plt.imshow(self.data.dpt)

    def draw_point(self):

        import cv2
        bg_img = np.ones(300*300).reshape(300,300)
        annoscale = 1
        joint = self.data.gtorig
        ax = bg_img
        hpe = self.hpe

        for i in range(len(hpe.jointConnections)):
            lc = tuple((hpe.jointConnectionColors[i]*255.).astype(int))
        #     lc = tuple((rgb_to_gray(self.jointConnectionColors[i])*255.).astype(int))
        #     lc = color
            cv2.line(ax, (int(numpy.rint(joint[hpe.jointConnections[i][0], 0])),
                          int(numpy.rint(joint[hpe.jointConnections[i][0], 1]))),
                     (int(numpy.rint(joint[hpe.jointConnections[i][1], 0])),
                      int(numpy.rint(joint[hpe.jointConnections[i][1], 1]))),
                     (102,0,0), thickness=3*annoscale, lineType=cv2.LINE_AA)
            plt.imshow(ax)

    def get_gtorig_shape(self):
        return self.data.gtcrop.shape
