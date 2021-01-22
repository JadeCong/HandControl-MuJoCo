# for mujoco mechanical hand MPL

from numpy.testing import assert_array_equal,assert_almost_equal
from mujoco_py import MjSim, MjViewer, load_model_from_xml, load_model_from_path, MjSimState, ignore_mujoco_warnings
import mujoco_py
import numpy as np
import cv2
from skimage.io import imsave, imshow
import matplotlib.pyplot as plt
import time
import json
from funtest.test_pathlib import first_try

import sys
sys.path.append("/home/jade/DRL/codes/MuJoCo/Hand_Control/")
from joint_position_calculator import get_pose_new6


xml_path = "/home/jade/DRL/codes/MuJoCo/xml_model/MPL/robot_hand.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
# sim.forward()


def print_hand_joint_pose(sim):
    print("hand pose:")
    print("wrist=", sim.data.get_joint_qpos("wrist_PRO"), sim.data.get_joint_qpos("wrist_UDEV"), sim.data.get_joint_qpos("wrist_FLEX"))
    print("thumb=", sim.data.get_joint_qpos("thumb_ABD"), sim.data.get_joint_qpos("thumb_MCP"), sim.data.get_joint_qpos("thumb_PIP"), sim.data.get_joint_qpos("thumb_DIP"))
    print("index=", sim.data.get_joint_qpos("index_ABD"), sim.data.get_joint_qpos("index_MCP"), sim.data.get_joint_qpos("index_PIP"), sim.data.get_joint_qpos("index_DIP"))
    print("middle=", sim.data.get_joint_qpos("middle_ABD"), sim.data.get_joint_qpos("middle_MCP"), sim.data.get_joint_qpos("middle_PIP"), sim.data.get_joint_qpos("middle_DIP"))
    print("ring=", sim.data.get_joint_qpos("ring_ABD"), sim.data.get_joint_qpos("ring_MCP"), sim.data.get_joint_qpos("ring_PIP"), sim.data.get_joint_qpos("ring_DIP"))
    print("pinky=", sim.data.get_joint_qpos("pinky_ABD"), sim.data.get_joint_qpos("pinky_MCP"), sim.data.get_joint_qpos("pinky_PIP"), sim.data.get_joint_qpos("pinky_DIP"))

    
print_hand_joint_pose(sim)

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

data_path = '/home/jade/DRL/datasets/picture/video_output/'
json_data = first_try(data_path, "VID*4734*.json")
it = iter(json_data)

while True:
    anyjson = str(next(it))
    print(anyjson)
    dt = json.load(open(anyjson))
    arr_right = np.array(dt['people'][0]['hand_right_keypoints_2d']).reshape(21,3)
    if arr_right.any():
        hand_pose = get_pose.get_hand_joint_pose(arr_right)
        print(hand_pose)

        sim_state = sim.get_state()

        sim_state.qpos[wrist_j0] = np.pi
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
        print("move hand to", hand_pose)
        print_hand_joint_pose(sim)
        viewer.render()
        time.sleep(0.001)
    else:
        continue