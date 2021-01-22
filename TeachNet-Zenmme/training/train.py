#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import argparse
import os
import time
import gc
import sys
sys.path.append("D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/")

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from dataset.datasets import MPLPairedDataset
from model.models import TeachingTeleModel, NaiveTeleModel

# set the garbage collection setting
gc.enable()
print("Now Garbage Collection Threshold: {}".format(gc.get_threshold()))
gc.set_threshold(10,2,2)
print("New Garbage Collection Threshold: {}".format(gc.get_threshold()))

# define parameter parser
parser = argparse.ArgumentParser(description='DeepMPLTeleoperation')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='../model/learned', help='pretrained model path')
parser.add_argument('--data-path', type=str, default='../dataset', help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=5)

args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available else False
if args.cuda:
    torch.cuda.manual_seed(1) #为当前GUP设置随机种子，使每次随机参数初始化的结果一致

# 设置训练日志以用于分析训练过程
logger = SummaryWriter(os.path.join('../evaluation/log/', args.tag))

np.random.seed(int(time.time()))

# 数据集加载器中的校对和初始化
def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# 设置数据集图片张数，尺寸，关节数，阈值精度，关节运动上下限
input_viewpoint=np.array([0,1,2,3,4,5,6,7,8])
input_size=100
embedding_size=128
joint_size=20
thresh_acc=[0.2, 0.25, 0.3] #TODO:figure out thresh_acc is what and the limits of joints
joint_upper_range = torch.tensor([2.07, 1.03, 1.03, 1.28, 
                                  0.345, 1.57, 1.72, 1.38,
                                  0.345, 1.57, 1.72, 1.38, 
                                  0.345, 1.57, 1.72, 1.38,
                                  0.345, 1.57, 1.72, 1.38]) # from thumb to pinky(from bottom to tip)
joint_lower_range = torch.tensor([0.0, 0.0, 0.0, -0.819,
                                  0.0, -0.785, 0.0, 0.0,
                                  0.0, -0.785, 0.0, 0.0,
                                  0.0, -0.785, 0.0, 0.0, 
                                  0.0, -0.785, 0.0, 0.0])

# 设置训练和测试的数据集加载器
train_loader = torch.utils.data.DataLoader(
    MPLPairedDataset(
        data_path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=True,
    ),
    batch_size=args.batch_size,
    num_workers=6,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

test_loader = torch.utils.data.DataLoader(
    MPLPairedDataset(
        data_path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=False,
        with_name=True,
    ),
    batch_size=args.batch_size,
    num_workers=6,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

# 设置恢复机制及模型选择
is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = TeachingTeleModel(input_size=input_size, embedding_size=embedding_size, joint_size=joint_size)
    # model = TeachingRENTeleModel(input_size=input_size, embedding_size=embedding_size, joint_size=joint_size)

# 设置使用GPU加载模型的运算方式
if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda() #单个GPU进行运算
    else:
        device_id = [1,2]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda() #多个GPU并行计算
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()

# 设置优化器和迭代器
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

# 设置最佳测试loss
# best_test_loss = np.inf

# 定义训练函数
def train(model, loader, epoch):
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    loss_mpl_reg = 0
    loss_mpl_cons = 0
    loss_human_reg = 0
    loss_human_cons = 0
    loss_align = 0
    train_error_mpl = 0
    train_error_human = 0
    correct_mpl, correct_human = [0,0,0], [0,0,0]
    for batch_idx, (mpl, human, target) in enumerate(loader):
        if args.cuda:
            mpl, human, target = mpl.cuda(), human.cuda(), target.cuda() #target为groundtruth和机械手的真实关节角度

        optimizer.zero_grad() #梯度归零

        # mpl part
        embedding_mpl, joint_mpl = model(mpl, is_human=False)
        #TODO: check if we need to reshape joint_upper_ranges, joint_lower_ranges
        joint_mpl = joint_mpl * (joint_upper_range - joint_lower_range) + joint_lower_range
        loss_mpl_reg = F.mse_loss(joint_mpl, target)
        loss_mpl_cons = constraints_loss(joint_mpl)/target.shape[0]
        loss_mpl = loss_mpl_reg + loss_mpl_cons

        # human part(因为robot mpl进行回归的准确率高，所以mpl为teacher，human为student)
        embedding_human, joint_human = model(human, is_human=True)
        joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range
        loss_human_reg = F.mse_loss(joint_human, target)
        loss_align = F.mse_loss(embedding_human, embedding_mpl.detach()) #consistency loss between human and mpl pose expression
        loss_human_cons = constraints_loss(joint_human)/target.shape[0]
        loss_human = loss_human_reg + loss_align + loss_human_cons

        # 计算总的loss并进行优化处理
        loss = loss_mpl + loss_human
        loss.backward()
        optimizer.step()

        loss = loss_mpl + loss_human

        # compute acc
        res_mpl = [np.sum(np.sum(abs(joint_mpl.cpu().data.numpy() - target.cpu().data.numpy()) < thresh, axis=-1) == joint_size) for thresh in thresh_acc]
        res_human = [np.sum(np.sum(abs(joint_human.cpu().data.numpy() - target.cpu().data.numpy()) < thresh, axis=-1) == joint_size) for thresh in thresh_acc]
        correct_mpl = [c + r for c, r in zip(correct_mpl, res_mpl)]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        train_error_mpl += F.l1_loss(joint_mpl, target, size_average=False)/joint_size
        train_error_human += F.l1_loss(joint_human, target, size_average=False)/joint_size

        # garbage collection
        gcc = gc.collect()
        # print("garbage object num: {}".format(gc.garbage))

        if batch_idx % args.log_interval == 0:
            if isinstance(loss_mpl_cons, float):
                loss_mpl_cons = torch.zeros(1)
            if isinstance(loss_human_cons, float):
                loss_human_cons = torch.zeros(1)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_reg_mpl: {:.6f}\t'
                    'Loss_cons_mpl: {:.6f}\tLoss_reg_human: {:.6f}\t'
                    'Loss_cons_human: {:.6f}\tLoss_align: {:.6f}\t{}'.format(
                    epoch, batch_idx * args.batch_size, len(loader.dataset),
                    100. * batch_idx * args.batch_size / len(loader.dataset),
                    loss.item(), loss_mpl_reg.item(), loss_mpl_cons.item(),
                    loss_human_reg.item(), loss_human_cons.item(), loss_align.item(), args.tag))

            logger.add_scalar('train_loss', loss.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_mpl_reg', loss_mpl_reg.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_mpl_cons', loss_mpl_cons.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_human_reg', loss_human_reg.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_human_cons', loss_human_cons.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_align', loss_align.item(),
                    batch_idx + epoch * len(loader))

    train_error_mpl /= len(loader.dataset)
    train_error_human /= len(loader.dataset)
    acc_mpl = [float(c) / float(len(loader.dataset)) for c in correct_mpl]
    acc_human = [float(c) / float(len(loader.dataset)) for c in correct_human]

    return acc_mpl, acc_human, train_error_mpl, train_error_human


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss_mpl_reg = 0
    test_loss_mpl_cons = 0
    test_loss_human_reg = 0
    test_loss_human_cons = 0
    test_loss_align = 0
    test_error_mpl = 0
    test_error_human = 0
    res = []
    correct_mpl, correct_human = [0,0,0], [0,0,0]
    for mpl, human, target, name in loader:
        if args.cuda:
            mpl, human, target = mpl.cuda(), human.cuda(), target.cuda()
        
        # mpl part
        embedding_mpl, joint_mpl = model(mpl, is_human=False)
        joint_mpl = joint_mpl * (joint_upper_range - joint_lower_range) + joint_lower_range
        test_loss_mpl_reg += F.mse_loss(joint_mpl, target, size_average=False).item()
        cons = constraints_loss(joint_mpl)
        if not isinstance(cons, float):
            test_loss_mpl_cons += cons 

        # human part
        embedding_human, joint_human = model(human, is_human=True)
        joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range
        test_loss_human_reg += F.mse_loss(joint_human, target, size_average=False).item()
        test_loss_align += F.mse_loss(embedding_human, embedding_mpl.detach(), size_average=False).item()
        cons = constraints_loss(joint_human)
        if not isinstance(cons, float):
            test_loss_human_cons += cons 

        # compute acc
        res_mpl = [np.sum(np.sum(abs(joint_mpl.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                      axis=-1) == joint_size) for thresh in thresh_acc]
        res_human = [np.sum(np.sum(abs(joint_human.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                      axis=-1) == joint_size) for thresh in thresh_acc]
        correct_mpl = [c + r for c, r in zip(correct_mpl, res_mpl)]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        test_error_mpl += F.l1_loss(joint_mpl, target, size_average=False)/joint_size
        test_error_human += F.l1_loss(joint_human, target, size_average=False)/joint_size
        res.append((name, joint_human))

    test_loss_mpl_reg /= len(loader.dataset)
    test_loss_mpl_cons /= len(loader.dataset)
    test_loss_human_reg /= len(loader.dataset)
    test_loss_align /= len(loader.dataset)
    test_loss_human_cons /= len(loader.dataset)
    test_loss = test_loss_mpl_reg + test_loss_human_reg + test_loss_align + test_loss_mpl_cons + test_loss_human_cons
    test_error_mpl /= len(loader.dataset)
    test_error_human /= len(loader.dataset)

    acc_mpl = [float(c)/float(len(loader.dataset)) for c in correct_mpl]
    acc_human = [float(c)/float(len(loader.dataset)) for c in correct_human]
    # f = open('input.csv', 'w')
    # for batch in res:
    #     for name, joint in zip(batch[0], batch[1]):
    #         buf = [name, '0.0', '0.0'] + [str(i) for i in joint.cpu().data.numpy()]
    #         f.write(','.join(buf) + '\n')

    return acc_mpl, acc_human, test_error_mpl, test_error_human, test_loss, test_loss_mpl_reg,\
           test_loss_mpl_cons, test_loss_human_reg, test_loss_human_cons, test_loss_align


# define physical loss for both human and mpl
def constraints_loss(joint_angle):
    loss_cons = 0.0
    F2_5_1 = [joint_angle[:, 4], joint_angle[:, 8], joint_angle[:, 12], joint_angle[:, 16]]
    F2_5_2 = [joint_angle[:, 5], joint_angle[:, 9], joint_angle[:, 13], joint_angle[:, 17]]
    F2_5_3 = [joint_angle[:, 6], joint_angle[:, 10], joint_angle[:, 14], joint_angle[:, 18]]
    F2_5_4 = [joint_angle[:, 7], joint_angle[:, 11], joint_angle[:, 15], joint_angle[:, 19]]
    F1_2_3 = [joint_angle[:, 1], joint_angle[:, 2]]

    for pos in F2_5_1:
        for f in pos:
            loss_cons = loss_cons + max(0.345 - f, 0) + max(f - 0.0, 0)
    for pos in F2_5_2:
        for f in pos:
            loss_cons = loss_cons + max(1.57 - f, 0) + max(f + 0.785, 0)
    for pos in F2_5_3:
        for f in pos:
            loss_cons = loss_cons + max(1.72 - f, 0) + max(f - 0.0, 0)
    for pos in F2_5_4:
        for f in pos:
            loss_cons = loss_cons + max(1.38 - f, 0) + max(f - 0.0, 0)
    for pos in F1_2_3:
        for f in pos:
            loss_cons = loss_cons + max(1.03 - f, 0) + max(f - 0.0, 0)
    for f in joint_angle[:, 0]:
        loss_cons = loss_cons + max(2.07 - f, 0) + max(f - 0.0, 0)
    for f in joint_angle[:, 3]:
        loss_cons = loss_cons + max(1.28 - f, 0) + max(f +0.819, 0)

    return loss_cons


def main():
    # global best_test_loss
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train_mpl, acc_train_human, train_error_mpl, train_error_human = train(model, train_loader, epoch)
            print('Train done, acc_mpl={}, acc_human={}, train_error_mpl={}, train_error_human={}'.format(acc_train_mpl, acc_train_human, train_error_mpl, train_error_human))
            acc_test_mpl, acc_test_human, test_error_mpl, test_error_human, loss, loss_mpl_reg, loss_mpl_cons, loss_human_reg,\
                    loss_human_cons, loss_align = test(model, test_loader)
            print('Test done, acc_mpl={}, acc_human={}, error_mpl ={}, error_human ={}, loss={}, loss_mpl_reg={}, loss_mpl_cons={}, '\
                  'loss_human_reg={}, loss_human_cons={}, loss_align={}'.format(acc_test_mpl, acc_test_human,
                                                                                 test_error_mpl, test_error_human,
                                                                                 loss, loss_mpl_reg, 
                                                                                 loss_mpl_cons, loss_human_reg, 
                                                                                 loss_human_cons, loss_align))
            logger.add_scalar('train_acc_mpl0.2', acc_train_mpl[0], epoch)
            logger.add_scalar('train_acc_mpl0.25', acc_train_mpl[1], epoch)
            logger.add_scalar('train_acc_mpl0.3', acc_train_mpl[2], epoch)
            logger.add_scalar('train_acc_human0.2', acc_train_human[0], epoch)
            logger.add_scalar('train_acc_human0.25', acc_train_human[1], epoch)
            logger.add_scalar('train_acc_human0.3', acc_train_human[2], epoch)

            logger.add_scalar('test_acc_mpl0.2', acc_test_mpl[0], epoch)
            logger.add_scalar('test_acc_mpl0.25', acc_test_mpl[1], epoch)
            logger.add_scalar('test_acc_mpl0.3', acc_test_mpl[2], epoch)
            logger.add_scalar('test_acc_human0.2', acc_test_human[0], epoch)
            logger.add_scalar('test_acc_human0.25', acc_test_human[1], epoch)
            logger.add_scalar('test_acc_human0.3', acc_test_human[2], epoch)

            logger.add_scalar('test_error_mpl', test_error_mpl, epoch)
            logger.add_scalar('test_error_human', test_error_human, epoch)

            logger.add_scalar('test_loss', loss, epoch)
            logger.add_scalar('test_loss_mpl_reg', loss_mpl_reg, epoch)
            logger.add_scalar('test_loss_mpl_cons', loss_mpl_cons, epoch)
            logger.add_scalar('test_loss_human_reg', loss_human_reg, epoch)
            logger.add_scalar('test_loss_align', loss_align, epoch)
            logger.add_scalar('test_loss_human_cons', loss_human_cons, epoch)

            #TODO: make sure the save the best model based on test_loss
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                # path = os.path.join(args.model_path, args.tag + '_best.model')
                torch.save(model, path)
                # is_best = loss < best_test_loss
                # best_test_loss = min(loss, best_test_loss)
                # if is_best:
                #     torch.save(model, path)
                print('Save model @ {}'.format(path))
                # print('Save best model @ {}'.format(path))
    else:
        print('testing...')
        acc_test_mpl, acc_test_human, test_error_mpl, test_error_human, loss, loss_mpl_reg, loss_mpl_cons, loss_human_reg, \
                loss_human_cons, loss_align = test(model, test_loader)
        print('Test done, acc_mpl={}, acc_human={}, error_mpl ={}, error_human ={}, loss={}, loss_mpl_reg={}, loss_mpl_cons={}, ' \
              'loss_human_reg={}, loss_human_cons={}, loss_align={},'.format(acc_test_mpl, acc_test_human,
                                                                           test_error_mpl, test_error_human,
                                                                           loss, loss_mpl_reg,
                                                                           loss_mpl_cons, loss_human_reg,
                                                                           loss_human_cons, loss_align))
if __name__ == "__main__":
    main()
