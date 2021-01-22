# for teleoperation test


#!/usr/bin/python
# -*- coding: utf-8 -*



"""
Control mechanical hand in virtual simulation environment(UE4/Unity/MuJoCo) or in real world scene(PR2/UR5) 
based the application protocol MQTT of IoT, this script works as a client node which connect to the broker 
agent for data processing and hand controller.
"""



import paho.mqtt.client as mqtt
import json
import time
import sys
sys.path.append("/home/jade/DRL/codes/MuJoCo/Hand_Control/drive_hand/")

from numpy.testing import assert_array_equal,assert_almost_equal
from mujoco_py import MjSim, MjViewer, load_model_from_xml, load_model_from_path, MjSimState, ignore_mujoco_warnings
import mujoco_py
import numpy as np
import cv2
from skimage.io import imsave, imshow
import matplotlib.pyplot as plt
from funtest.test_pathlib import first_try



xml_path = "/home/jade/DRL/codes/MuJoCo/Resources/xml_model/MPL/robot_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

wrist_j0 = sim.model.get_joint_qpos_addr("wrist_PRO")
wrist_j1 = sim.model.get_joint_qpos_addr("wrist_UDEV")
wrist_j2 = sim.model.get_joint_qpos_addr("wrist_FLEX")

thumb_j0 = sim.model.get_joint_qpos_addr("thumb_ABD")
thumb_j1 = sim.model.get_joint_qpos_addr("thumb_MCP")
thumb_j2 = sim.model.get_joint_qpos_addr("thumb_PIP")
thumb_j3 = sim.model.get_joint_qpos_addr("thumb_DIP")

index_j0 = sim.model.get_joint_qpos_addr("index_ABD")
index_j1 = sim.model.get_joint_qpos_addr("index_MCP")
index_j2 = sim.model.get_joint_qpos_addr("index_PIP")
index_j3 = sim.model.get_joint_qpos_addr("index_DIP")

middle_j0 = sim.model.get_joint_qpos_addr("middle_ABD")
middle_j1 = sim.model.get_joint_qpos_addr("middle_MCP")
middle_j2 = sim.model.get_joint_qpos_addr("middle_PIP")
middle_j3 = sim.model.get_joint_qpos_addr("middle_DIP")

ring_j0 = sim.model.get_joint_qpos_addr("ring_ABD")
ring_j1 = sim.model.get_joint_qpos_addr("ring_MCP")
ring_j2 = sim.model.get_joint_qpos_addr("ring_PIP")
ring_j3 = sim.model.get_joint_qpos_addr("ring_DIP")

pinky_j0 = sim.model.get_joint_qpos_addr("pinky_ABD")
pinky_j1 = sim.model.get_joint_qpos_addr("pinky_MCP")
pinky_j2 = sim.model.get_joint_qpos_addr("pinky_PIP")
pinky_j3 = sim.model.get_joint_qpos_addr("pinky_DIP")



'''
construct the MQTT client for communicating with MQTT broker and realizing data transmission
'''
class MQTTClient:
    def __init__(self, client_id="MQTT_Client", broker_host="192.168.6.131", broker_port=1883, broker_user=None, broker_pass=None, client_keepalive=60):
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
        # self.msg_sub = json.dumps({"thumb": "[0.0, 0.0, 0.0, 0.0]", "index": "[0.0, 0.0, 0.0, 0.0]", "middle": "[0.0, 0.0, 0.0, 0.0]", "ring": "[0.0, 0.0, 0.0, 0.0]", "pinky": "[0.0, 0.0, 0.0, 0.0]"})
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
        self.connect_async = self.client.connect_async #asynchronous connect the broker in a non-blocking manner, and the the connection will not complete until loop_start() is called
        self.connect_srv = self.client.connect_srv #connect a broker using an SRV DNS lookup to obtain the broker address
        self.loop = self.client.loop
        self.loop_start = self.client.loop_start
        self.loop_stop = self.client.loop_stop
        self.loop_forever = self.client.loop_forever #blocking network and will not return until the client calls disconnect()


    def connect(self):
        self.client.connect(self.broker_host, self.broker_port, self.client_keepalive)
        print("Connecting to server[%s]......"%self.broker_host)


    def on_connect(self, client, userdata, flags, rc):
        print("Connected to sever[%s] with result code "%self.broker_host + str(rc))


    def disconnect(self):
        self.client.disconnect()
        print("Disconnecting from server[%s]......"%self.broker_host)
    

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from server[%s] with result code "%self.broker_host + str(rc))
    

    # @staticmethod
    def subscribe(self):
        self.client.subscribe(self.topic_sub, self.qos_sub)
        print("Subscribing topic[%s] from server[%s]......"%(self.topic_sub, self.broker_host))


    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed topic[%s] form server[%s]: "%(self.topic_sub, self.broker_host) + str(mid) + " " + str(granted_qos))


    def unsubscribe(self):
        self.client.unsubscribe(self.topic_sub)
        print("Unsubscribing topic[%s] from server[%s]......"%(self.topic_sub, self.broker_host))
    

    def on_unsubscribe(self, client, userdata, mid):
        print("Unsubscribed topic[%s] from server[%s]: "%(self.topic_sub, self.broker_host) + str(mid) + " " + str(self.qos_sub))    
    # @staticmethod
    def publish(self):
        self.client.publish(self.topic_pub, payload=self.msg_pub, qos=self.qos_pub, retain=False)
        print("Publishing topic[%s] to server[%s]......"%(self.topic_pub, self.broker_host))


    def on_publish(self, client, userdata, mid):
        print("Published topic[%s] to server[%s]: "%(self.topic_pub, self.broker_host) + str(mid))


    def on_message(self, client, userdata, msg):
        try:
            self.msg_sub=json.loads(msg.payload)
            print("Received msg[%s] from server[%s]"%(self.msg_sub, self.broker_host))
        except json.JSONDecodeError as e:
            print("Parsing Msg Error: %s"%e.msg)
    

    def on_log(self, client, userdata, level, string):
        print(string)


        
def print_hand_joint_pose(sim):
    print("hand pose:")
    print("wrist=", sim.data.get_joint_qpos("wrist_PRO"), sim.data.get_joint_qpos("wrist_UDEV"), sim.data.get_joint_qpos("wrist_FLEX"))
    print("thumb=", sim.data.get_joint_qpos("thumb_ABD"), sim.data.get_joint_qpos("thumb_MCP"), sim.data.get_joint_qpos("thumb_PIP"), sim.data.get_joint_qpos("thumb_DIP"))
    print("index=", sim.data.get_joint_qpos("index_ABD"), sim.data.get_joint_qpos("index_MCP"), sim.data.get_joint_qpos("index_PIP"), sim.data.get_joint_qpos("index_DIP"))
    print("middle=", sim.data.get_joint_qpos("middle_ABD"), sim.data.get_joint_qpos("middle_MCP"), sim.data.get_joint_qpos("middle_PIP"), sim.data.get_joint_qpos("middle_DIP"))
    print("ring=", sim.data.get_joint_qpos("ring_ABD"), sim.data.get_joint_qpos("ring_MCP"), sim.data.get_joint_qpos("ring_PIP"), sim.data.get_joint_qpos("ring_DIP"))
    print("pinky=", sim.data.get_joint_qpos("pinky_ABD"), sim.data.get_joint_qpos("pinky_MCP"), sim.data.get_joint_qpos("pinky_PIP"), sim.data.get_joint_qpos("pinky_DIP"))

        
    
"""
Construct the instance for the MQTT Client and communicate with the broker server for data transmission
"""
# instance the mqtt client and connect to the broker server
mqtt_client = MQTTClient("Jade-MuJoCo")
mqtt_client.connect()

# redefine mqtt client topic_pub/topic_sub parameters
mqtt_client.topic_pub = "ue4_states"
mqtt_client.qos_pub = 0
mqtt_client.topic_sub = "hand_pose"
mqtt_client.qos_sub = 0


# hand_pose_zero = {"thumb": [1.0, 0.0, 0.0, 0.0], "index": [1.0, 0.0, 0.0, 0.0], "middle": [0.0, 0.0, 0.0, 0.0], "ring": [0.0, 0.0, 0.0, 0.0], "pinky": [0.0, 0.0, 0.0, 0.0]}

try:
    mqtt_client.subscribe()   

    while True:
        mqtt_client.loop_start()
        
        hand_pose = mqtt_client.msg_sub
        # print(hand_pose)
        
        if hand_pose==None:
            continue
        else:
            print("haha")

            sim_state = sim.get_state()

            sim_state.qpos[wrist_j0] = 0.0
            sim_state.qpos[wrist_j1] = 0.0
            sim_state.qpos[wrist_j2] = 0.0

            sim_state.qpos[thumb_j0] = hand_pose['thumb'][0]
            sim_state.qpos[thumb_j1] = hand_pose['thumb'][1]
            sim_state.qpos[thumb_j2] = hand_pose['thumb'][2]
            sim_state.qpos[thumb_j3] = hand_pose['thumb'][3]

            sim_state.qpos[index_j0] = hand_pose['index'][0]
            sim_state.qpos[index_j1] = hand_pose['index'][1]
            sim_state.qpos[index_j2] = hand_pose['index'][2]
            sim_state.qpos[index_j3] = hand_pose['index'][3]

            sim_state.qpos[middle_j0] = hand_pose['middle'][0]
            sim_state.qpos[middle_j1] = hand_pose['middle'][1]
            sim_state.qpos[middle_j2] = hand_pose['middle'][2]
            sim_state.qpos[middle_j3] = hand_pose['middle'][3]

            sim_state.qpos[ring_j0] = hand_pose['ring'][0]
            sim_state.qpos[ring_j1] = hand_pose['ring'][1]
            sim_state.qpos[ring_j2] = hand_pose['ring'][2]
            sim_state.qpos[ring_j3] = hand_pose['ring'][3]

            sim_state.qpos[pinky_j0] = hand_pose['pinky'][0]
            sim_state.qpos[pinky_j1] = hand_pose['pinky'][1]
            sim_state.qpos[pinky_j2] = hand_pose['pinky'][2]
            sim_state.qpos[pinky_j3] = hand_pose['pinky'][3]

            sim.set_state(sim_state)

            sim.forward()
            sim.step()

            viewer.render()
#             time.sleep(0.1)

finally:
    mqtt_client.disconnect()
    print("what the fuck")

