# for another mujoco mechanical hand Adroit

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


xml_path = "/home/jade/DRL/codes/MuJoCo/xml_model/Adroit/Adroit_hand_withOverlay.xml"
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
# sim.forward()


def print_hand_joint_pose(sim):
    print("hand pose:")
    print("wrist=", sim.data.get_joint_qpos("WRJ1"), sim.data.get_joint_qpos("WRJ0"))
    print("thumb=", sim.data.get_joint_qpos("THJ4"), sim.data.get_joint_qpos("THJ3"), sim.data.get_joint_qpos("THJ2"), sim.data.get_joint_qpos("THJ1"), sim.data.get_joint_qpos("THJ0"))
    print("index=", sim.data.get_joint_qpos("FFJ3"), sim.data.get_joint_qpos("FFJ2"), sim.data.get_joint_qpos("FFJ1"), sim.data.get_joint_qpos("FFJ0"))
    print("middle=", sim.data.get_joint_qpos("MFJ3"), sim.data.get_joint_qpos("MFJ2"), sim.data.get_joint_qpos("MFJ1"), sim.data.get_joint_qpos("MFJ0"))
    print("ring=", sim.data.get_joint_qpos("RFJ3"), sim.data.get_joint_qpos("RFJ2"), sim.data.get_joint_qpos("RFJ1"), sim.data.get_joint_qpos("RFJ0"))
    print("pinky=", sim.data.get_joint_qpos("LFJ4"), sim.data.get_joint_qpos("LFJ3"), sim.data.get_joint_qpos("LFJ2"), sim.data.get_joint_qpos("LFJ1"), sim.data.get_joint_qpos("LFJ0"))

    
print_hand_joint_pose(sim)

wrist_j0 = sim.model.get_joint_qpos_addr("WRJ1")
wrist_j1 = sim.model.get_joint_qpos_addr("WRJ0")

thumb_j0 = sim.model.get_joint_qpos_addr("THJ4")
thumb_j1 = sim.model.get_joint_qpos_addr("THJ3")
thumb_j2 = sim.model.get_joint_qpos_addr("THJ2")
thumb_j3 = sim.model.get_joint_qpos_addr("THJ1")
thumb_j4 = sim.model.get_joint_qpos_addr("THJ0")

index_j0 = sim.model.get_joint_qpos_addr("FFJ3")
index_j1 = sim.model.get_joint_qpos_addr("FFJ2")
index_j2 = sim.model.get_joint_qpos_addr("FFJ1")
index_j3 = sim.model.get_joint_qpos_addr("FFJ0")

middle_j0 = sim.model.get_joint_qpos_addr("MFJ3")
middle_j1 = sim.model.get_joint_qpos_addr("MFJ2")
middle_j2 = sim.model.get_joint_qpos_addr("MFJ1")
middle_j3 = sim.model.get_joint_qpos_addr("MFJ0")

ring_j0 = sim.model.get_joint_qpos_addr("RFJ3")
ring_j1 = sim.model.get_joint_qpos_addr("RFJ2")
ring_j2 = sim.model.get_joint_qpos_addr("RFJ1")
ring_j3 = sim.model.get_joint_qpos_addr("RFJ0")

pinky_j0 = sim.model.get_joint_qpos_addr("LFJ4")
pinky_j1 = sim.model.get_joint_qpos_addr("LFJ3")
pinky_j2 = sim.model.get_joint_qpos_addr("LFJ2")
pinky_j3 = sim.model.get_joint_qpos_addr("LFJ1")
pinky_j4 = sim.model.get_joint_qpos_addr("LFJ0")

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

        sim_state.qpos[wrist_j0] = 0.0
        sim_state.qpos[wrist_j1] = 0.0
        
        sim_state.qpos[thumb_j0] = 0.0
        sim_state.qpos[thumb_j1] = - hand_pose['thumb'][0]
        sim_state.qpos[thumb_j2] = - hand_pose['thumb'][1]
        sim_state.qpos[thumb_j3] = - hand_pose['thumb'][2]
        sim_state.qpos[thumb_j4] = - hand_pose['thumb'][3]
        
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

        sim_state.qpos[pinky_j0] = 0.0
        sim_state.qpos[pinky_j1] = hand_pose['pinky'][0]
        sim_state.qpos[pinky_j2] = hand_pose['pinky'][1]
        sim_state.qpos[pinky_j3] = hand_pose['pinky'][2]
        sim_state.qpos[pinky_j4] = hand_pose['pinky'][3]

        sim.set_state(sim_state)

        sim.forward()
        sim.step()
        print("move hand to", hand_pose)
        print_hand_joint_pose(sim)
        viewer.render()
        time.sleep(0.001)
    else:
        continue