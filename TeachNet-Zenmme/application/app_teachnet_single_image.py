#!/usr/bin/env python
# -*- coding:UTF-8 -*-



# the description of this script which is necessary
"""
This script is for test, and we can use the learned model to generate joint angles
for controlling mechanical hand in MuJoCo/UE4/Real-World based on the pictures or
video frames of any human hand.
"""



# import the dependencies modules and self-defined packages
import sys
sys.path.append("D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/")
import os
import time
import gc
import argparse

import torch
import torch.utils.data
import torch.nn as nn

import numpy as np
import pyrealsense2 as rs
import cv2
import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt



# set the python garbage collection setting for avoiding oom
gc.enable()
print("Now Garbage Collection Threshold: {}".format(gc.get_threshold()))
gc.set_threshold(20,2,2)
print("New Garbage Collection Threshold: {}".format(gc.get_threshold()))

# define parameter parser and parse the user's specific parameters
parser = argparse.ArgumentParser(description='DeepTeleoperation:MuJoCo-MPL')
parser.add_argument('--tag', type=str, default='mujoco-mpl')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model-path', type=str, default='../model/learned/teachnet-zenmme_65.model', help='learned model path')
parser.add_argument('--data-path', type=str, default='../dataset', help='data path')
args = parser.parse_args()

# set the GPU process speed acceleration based on whether the cuda is avaiable
args.cuda = args.cuda if torch.cuda.is_available else False
if args.cuda:
    #为当前GUP设置随机种子，使每次随机参数初始化的结果一致（即生成的随机数一样），若有多个GPU，则manual_seed_all()为所有GUP设置种子
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time())) #基于时间戳每次产生不同的随机数

# set the global variables and parameters(network parameters and joint limits)
input_size = 100
embedding_size = 128
joint_size = 20
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

# load the model and send them into GPU
# model = torch.load(args.model_path, map_location='cuda:{}'.format(args.gpu)) # or map to cpu
model = torch.load(args.model_path, map_location='cpu')
model.device_ids = [args.gpu]
print('load model {}'.format(args.model_path))

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu) #指定GPU设备列表中的起始编号
        model = model.cuda() #指定使用某一块GPU
    else:
        device_id = [0] #指定要使用的多块GPU的编号列表
        torch.cuda.set_device(device_id[0]) #指定GPU设备列表中的起始编号
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda() #须torch.tensor()格式的数据才能送进GPU进行处理(如上)
    joint_lower_range = joint_lower_range.cuda()



# define related class which can be used in this script
class MQTTClient:
    def __init__(self, client_id="MQTT_Client", broker_host="192.168.6.131", broker_port=1883, broker_user=None,
                 broker_pass=None, client_keepalive=60):
        # define mqtt parameters
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.broker_user = broker_user
        self.broker_pass = broker_pass
        self.client_keepalive = client_keepalive

        # redefine the topic parameters while dealing with topic
        self.topic_sub = "msg_from_server"
        self.qos_sub = 0
        self.msg_sub = None
        self.topic_pub = "msg_to_server"
        self.qos_pub = 0
        self.msg_pub = None

        # overwrite mqtt functions for specified usage
        self.client = mqtt.Client(self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message
        self.client.on_log = self.on_log

        # change connection method and loop method
        self.connect_async = self.client.connect_async  # asynchronous connect the broker in a non-blocking manner, and the the connection will not complete until loop_start() is called
        self.connect_srv = self.client.connect_srv  # connect a broker using an SRV DNS lookup to obtain the broker address
        self.loop = self.client.loop
        self.loop_start = self.client.loop_start
        self.loop_stop = self.client.loop_stop
        self.loop_forever = self.client.loop_forever  # blocking network and will not return until the client calls disconnect()

    def connect(self):
        self.client.connect(self.broker_host, self.broker_port, self.client_keepalive)
        print("Connecting to server[%s]......" % self.broker_host)

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to sever[%s] with result code " % self.broker_host + str(rc))

    def disconnect(self):
        self.client.disconnect()
        print("Disconnecting from server[%s]......" % self.broker_host)

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from server[%s] with result code " % self.broker_host + str(rc))

    # @staticmethod
    def subscribe(self):
        self.client.subscribe(self.topic_sub, self.qos_sub)
        print("Subscribing topic[%s] from server[%s]......" % (self.topic_sub, self.broker_host))

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed topic[%s] form server[%s]: " % (self.topic_sub, self.broker_host) + str(mid) + " " + str(
            granted_qos))

    def unsubscribe(self):
        self.client.unsubscribe(self.topic_sub)
        print("Unsubscribing topic[%s] from server[%s]......" % (self.topic_sub, self.broker_host))

    def on_unsubscribe(self, client, userdata, mid):
        print("Unsubscribed topic[%s] from server[%s]: " % (self.topic_sub, self.broker_host) + str(mid) + " " + str(
            self.qos_sub))

    # @staticmethod
    def publish(self):
        self.client.publish(self.topic_pub, payload=self.msg_pub, qos=self.qos_pub, retain=False)
        print("Publishing topic[%s] to server[%s]......" % (self.topic_pub, self.broker_host))

    def on_publish(self, client, userdata, mid):
        print("Published topic[%s] to server[%s]: " % (self.topic_pub, self.broker_host) + str(mid))

    def on_message(self, client, userdata, msg):
        try:
            self.msg_sub = json.loads(msg.payload)
            print("Received msg[%s] from server[%s]" % (self.msg_sub, self.broker_host))
        except json.JSONDecodeError as e:
            print("Parsing Msg Error: %s" % e.msg)

    def on_log(self, client, userdata, level, string):
        print(string)


class DeepTeleoperation(object):
    def __init__(self):
        # self.start = {'thumb': [0.0, 0.0, 0.0, 0.0],
        #               'index': [0.0, 0.0, 0.0, 0.0],
        #               'middle': [0.0, 0.0, 0.0, 0.0],
        #               'ring': [0.0, 0.0, 0.0, 0.0],
        #               'pinky': [0.0, 0.0, 0.0, 0.0]}

        self.mqtt_client = MQTTClient("Jade-Tele")
        self.mqtt_client.topic_pub = "hand_pose"
        self.mqtt_client.qos_pub = 0
        self.mqtt_client.topic_sub = "ue4_states"
        self.mqtt_client.qos_sub = 0
        self.mqtt_client.connect()
        self.pic_path = "D:/Tempfiles/PycharmProjects/DRL/codes/MuJoCo/Hand_Control/TeachNet-Zenmme/dataset/human/human_crop/"
        self.pic_name = os.listdir(self.pic_path)
        self.pic_index = 0

    def tele_callback(self, image):
        # get the image
        # img = cv2.imread(os.path.join(self.pic_path, self.pic_name[self.pic_index]), -1)
        # plt.imshow(img)
        # plt.show()

        # get the joint angles
        goal = self.joint_cal(image)
        # print(type(goal))
        # print(goal)

        # send the joints to broker through mqtt
        self.mqtt_client.msg_pub = json.dumps(goal)
        print(type(self.mqtt_client.msg_pub))
        print(self.mqtt_client.msg_pub)
        self.mqtt_client.publish()

    def joint_cal(self, image):
        # run the model
        global model
        featrue = test(model, image)
        joint = featrue.tolist()

        # crop joints
        joint[0] = self.clip(joint[0], 2.07, 0.0)
        joint[1] = self.clip(joint[1], 1.03, 0.0)
        joint[2] = self.clip(joint[2], 1.03, 0.0)
        joint[3] = self.clip(joint[3], 1.28, -0.819)
        joint[4] = self.clip(joint[4], 0.345, 0.0)
        joint[5] = self.clip(joint[5], 1.57, -0.785)
        joint[6] = self.clip(joint[6], 1.72, 0.0)
        joint[7] = self.clip(joint[7], 1.38, 0.0)
        joint[8] = self.clip(joint[8], 0.345, 0.0)
        joint[9] = self.clip(joint[9], 1.57, -0.785)
        joint[10] = self.clip(joint[10], 1.72, 0.0)
        joint[11] = self.clip(joint[11], 1.38, 0.0)
        joint[12] = self.clip(joint[12], 0.345, 0.0)
        joint[13] = self.clip(joint[13], 1.57, -0.785)
        joint[14] = self.clip(joint[14], 1.72, 0.0)
        joint[15] = self.clip(joint[15], 1.38, 0.0)
        joint[16] = self.clip(joint[16], 0.345, 0.0)
        joint[17] = self.clip(joint[17], 1.57, -0.785)
        joint[18] = self.clip(joint[18], 1.72, 0.0)
        joint[19] = self.clip(joint[19], 1.38, 0.0)

        # make a dict
        hand_pose = dict([("thumb", joint[0:4]), ("index", joint[4:8]),
                          ("middle", joint[8:12]), ("ring", joint[12:16]),
                          ("pinky", joint[16:20])])
        # hand_pose.update(thumb = joint[0:4])
        # hand_pose.update(index = joint[4:8])
        # hand_pose.update(middle = joint[8:12])
        # hand_pose.update(ring = joint[12:16])
        # hand_pose.update(pinky = joint[16:20])

        return hand_pose

    def clip(self, x, maxv=None, minv=None):
        if maxv is not None and x > maxv:
            x = maxv
        if minv is not None and x < minv:
            x = minv
        return x



# define the functions which can be used in this script
def test(model, image):
    # set the model running mode
    model.eval()
    torch.set_grad_enabled(False)

    # check the input image size
    assert(image.shape == (input_size, input_size)), "Wrong size for input depth image(100x100)!"
    # assert (image.size == (input_size, input_size)), "Wrong size for input depth image(100x100)!"
    image = image[np.newaxis, np.newaxis, ...]
    image = torch.Tensor(image)
    if args.cuda:
        image = image.cuda()

    # get the human hand joint angle
    embedding_human, joint_human = model(image, is_human=True)
    joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range

    return joint_human.cpu().data.numpy()[0]


def main():
    print('Let us go for teleoperating mpl in mujoco...')
    tele_mpl = DeepTeleoperation()

    # Configure depth and color streams
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    # pipeline.start(config)
    file_path = "D:\Tempfiles\PycharmProjects\DRL\codes\MuJoCo\Hand_Control\TeachNet-Zenmme\dataset\human\human_crop"
    file_name = os.listdir("D:\Tempfiles\PycharmProjects\DRL\codes\MuJoCo\Hand_Control\TeachNet-Zenmme\dataset\human\human_crop")

    try:
        while True:
            # command = input("manual input:")
            # if command == 'q' or command == 'Q':
            #     print("The teleoperation is ended by manual...")
            #     break
            # else:
            # frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # if not depth_frame:
            #     continue
            #
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_max = np.max(depth_image)
            # depth_min = np.min(depth_image)
            # depth_scale = (depth_max - depth_min) / 255.0
            # depth_image = (depth_image - depth_min) / depth_scale
            #
            # depth_image1 = depth_image[int(320 - 50):int(320 + 50), int(240 - 50):int(240 + 50)]
            # print("depth_image size:{}".format(np.shape(depth_image)))

            file_index = np.random.randint(0,2)
            depth_image1 = cv2.imread(os.path.join(file_path, file_name[file_index]), cv2.IMREAD_UNCHANGED)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', depth_image)
            cv2.imshow('RealSense Crop', depth_image1)
            key = cv2.waitKey(1)

            tele_mpl.pic_index = np.random.randint(0, 70000)
            tele_mpl.tele_callback(depth_image1)
            # time.sleep(0.5)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # pipeline.stop()
        tele_mpl.disconnect()



# define the main function entry
if __name__ == "__main__":
    main()

