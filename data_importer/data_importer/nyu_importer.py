import numpy
import gc
import matplotlib
matplotlib.use("Agg")
import sys
from PIL import Image
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
# from net.scalenet import ScaleNetParams, ScaleNet
# from trainer.scalenettrainer import ScaleNetTrainerParams, ScaleNetTrainer
from data_importer.util.handdetector import HandDetector
import os
import numpy as np

from data_importer.data.importers import NYUImporter
from data_importer.data.dataset import NYUDataset

from data_importer.util.helpers import shuffle_many_inplace
from data_importer.util.handpose_evaluation import ICVLHandposeEvaluation, NYUHandposeEvaluation, MSRAHandposeEvaluation

from pathlib import Path
CURRENT_DIR = Path(__file__).parent

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

        dataset_path = os.path.join(str(CURRENT_DIR.parent),"data/NYU")
        cache_path = os.path.join(str(CURRENT_DIR),"cache")
        self.di = NYUImporter(dataset_path,cacheDir=cache_path)
        # Seq1_1 = di.loadSequence('train', shuffle=True, rng=rng, docom=False)
        # Seq1_1 = Seq1_1._replace(name='train_gt')
        Seq1_2 = self.di.loadSequence('train', shuffle=True, rng=rng, docom=True)
        Seq1_2 = Seq1_2._replace(name='train_com')
        # trainSeqs = [Seq1_1, Seq1_2]
        seqs= [Seq1_2]

        # self.hpe = NYUHandposeEvaluation(gt3D, joints)

        if ix:
            sample_ix= ix
        else:
            sample_ix= np.random.randint(1000)
            print(sample_ix)

        self.data=seqs[0].data[sample_ix]
        # self.get_origin_img(self.data.fileName)
        print("you now choosing seq 0 {}".format(sample_ix))
        plt.imshow(self.data.dpt)

    def draw_point(self,step=100):

        import cv2
        bg_img = np.ones(300*300).reshape(300,300)
        annoscale = 1
        joint = self.data.gtorig
        ax = bg_img
        hpe = self.hpe

        for i in range(len(hpe.jointConnections)):
        # for i in range(step):
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

    def get_origin_img(self):

        from skimage.io import imshow

        self.orig_dept=self.di.loadDepthMap(self.data.fileName)
        imshow(self.orig_dept)

def get_dept(filename):

    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """

    from skimage.io import imshow
    import cv2
    img = Image.open(filename)
    # top 8 bits of depth are packed into green channel and lower 8 bits into blue
    assert len(img.getbands()) == 3
    # r, g, b = img.split()
    # r = np.asarray(r, np.int32)
    # g = np.asarray(g, np.int32)
    # b = np.asarray(b, np.int32)
    # dpt = np.bitwise_or(np.left_shift(g, 8), b)
    # imgdata = np.asarray(img, np.float32)
    # im = Image.fromarray(imgdata)
    # im.save("./test.png")
    # img.show()
    # fig = plt.figure()
    # plt.show(imgdata)
    # fig.savefig("./test.png")
    # imshow(imgdata)
    img  = cv2.imread(filename, -1)
    b_channel, g_channel, r_channel = cv2.split(img)
    imgdata = cv2.merge((g_channel, b_channel,r_channel))
    cv2.imshow("img",imgdata)
    cv2.waitKey(2)
    # cv2.destroyAllWindows()
    return imgdata

def get_dept_cv2(filename):
    import cv2

    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    img  = cv2.imread(filename, -1)
    depth=cv2.split(img)[0]
    depth[depth>800]=0
    depth=depth/1000.0000
    cv2.imshow('img',depth)
    cv2.waitKey(1)
    return depth
def plot_line():
    a=[1,3,4]
    b=[1,3,4]
    fig = plt.figure()
    plt.plot(a,b)
    plt.show()
    fig.savefig("test.png")

