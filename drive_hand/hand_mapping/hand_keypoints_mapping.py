# This file is used for mapping the human hand keypoints(ground truth) to the mechanical hand(MPL/Adroit) keypoints in MuJoCo based on the link length scale.

import os
import numpy as np
from pyquaternion import Quaternion


def calc_mpl_finger_length(PIP, DIP, site):
    # calc the last three finger link length for mpl every finger(thumb, index, middle, ring, pinky)
    link2 = calc_vector_length(PIP)
    link3 = calc_vector_length(DIP)
    # link4 = calc_vector_length(site)
    link4 = link3 - 0.005

    mpl_finger_length = np.array([link2, link3, link4])

    return mpl_finger_length

def quaternion_to_axisangle(quaternion):
    angle = 2 * np.arccos(quaternion[0])
    x = quaternion[1] / np.sin(angle / 2)
    y = quaternion[2] / np.sin(angle / 2)
    z = quaternion[3] / np.sin(angle / 2)
    axis = np.array([x, y, z])

    return axis, angle


def coordinate_transform(new_pos, ref_pos, ref_quat):
    # axis1, angle1 = quaternion_to_axisangle(ref_quat)
    # quaternion = Quaternion(axis=axis1, angle=angle1)
    quaternion = Quaternion(ref_quat[0], ref_quat[1], ref_quat[2], ref_quat[3])
    origin_pos = quaternion.rotate(new_pos) + ref_pos

    return origin_pos


def calc_first_link_length(MCP_pos, ABD_pos, ABD_quat):
    # calc the first link length of the each finger
    mcp_to_palm_pos = coordinate_transform(MCP_pos, ABD_pos, ABD_quat)
    link1 = calc_vector_length(mcp_to_palm_pos)

    return link1


def calc_model_mpl_fingers_length():
    # the model MPL finger length unit:meter(m)
    # step 1: get the position of finger joints in the world reference coordinate(five fingers at the same level)
    # for the whole robot MPL
    mpl_pos = np.array([0, -0.35, 0.2]) #mpl hand model relative to the world coordinate
    mpl_axisangle = np.array([0, 0, 1, 3.141592])
    # for the wrist
    wristy = np.array([0, 0, 0]) #wrist_PRO
    wristx = np.array([-3.36826e-005, -0.0476452, 0.00203763]) #wrist_UDEV
    wristz = np.array([0.0001872, -0.03, -0.002094]) #wrist_FLEX
    # for thumb finger
    palm = np.array([0.025625, 0, 0])
    thumb_ABD = np.array([0.00835752, -0.0206978, -0.010093])
    thumb_ABD_quat = np.array([0.733, 0.462, -0.191, -0.462])
    # thumb_ABD_quat = np.array([0.990237, 0.0412644, -0.0209178, -0.13149])
    thumb_MCP = np.array([0.0209172, -0.00084, 0.0014476])
    thumb_PIP = np.array([0.0335, 0, -0.0007426])
    thumb_DIP = np.array([0.0335, 0, 0.0010854])
    thumb_site = np.array([0.0156, -0.007, 0.0003])
    # for index finger
    index_ABD = np.array([0.00986485, -0.0658, 0.00101221])
    index_ABD_quat = np.array([0.996195, 0, 0.0871557, 0])
    index_MCP = np.array([6.26e-005, -0.018, 0])
    index_PIP = np.array([0.001086, -0.0435, 0.0005])
    index_DIP = np.array([-0.000635, -0.0245, 0])
    index_site = np.array([0, -0.0132, -0.0038])
    # for middle finger
    middle_ABD = np.array([-0.012814, -0.0779014, 0.00544608])
    middle_ABD_quat = np.array([-3.14, 0.0436194, 0, 0])
    middle_MCP = np.array([6.26e-005, -0.018, 0])
    middle_PIP = np.array([0.001086, -0.0435, 0.0005])
    middle_DIP = np.array([-0.000635, -0.0245, 0])
    middle_site = np.array([0, -0.0129, -0.0038])
    # for ring finger
    ring_ABD = np.array([-0.0354928, -0.0666999, 0.00151221])
    ring_ABD_quat = np.array([0.996195, 0, -0.0871557, 0])
    ring_MCP = np.array([6.26e-005, -0.018, 0])
    ring_PIP = np.array([0.001086, -0.0435, 0.0005])
    ring_DIP = np.array([-0.000635, -0.0245, 0])
    ring_site = np.array([0, -0.0117, -0.0038])
    # for pinky finger
    pinky_ABD = np.array([-0.0562459, -0.0554001, -0.00563858])
    pinky_ABD_quat = np.array([0.996195, 0, -0.0871557, 0])
    pinky_MCP = np.array([6.26e-005, -0.0178999, 0])
    pinky_PIP = np.array([0.000578, -0.033, 0.0005])
    pinky_DIP = np.array([-4.78e-005, -0.0175, 0])
    pinky_site = np.array([0, -0.0121, -0.0038])

    # step 2: calc the fingers length
    ## calc the last three link length of five fingers
    thumb_length = calc_mpl_finger_length(thumb_PIP, thumb_DIP, thumb_site)
    index_length = calc_mpl_finger_length(index_PIP, index_DIP, index_site)
    middle_length = calc_mpl_finger_length(middle_PIP, middle_DIP, middle_site)
    ring_length = calc_mpl_finger_length(ring_PIP, ring_DIP, ring_site)
    pinky_length = calc_mpl_finger_length(pinky_PIP, pinky_DIP, pinky_site)
    
    ## calc the first link length of five fingers
    thumb_link1 = calc_first_link_length(thumb_MCP, thumb_ABD, thumb_ABD_quat)
    index_link1 = calc_first_link_length(index_MCP, index_ABD, index_ABD_quat)
    middle_link1 = calc_first_link_length(middle_MCP, middle_ABD, middle_ABD_quat)
    ring_link1 = calc_first_link_length(ring_MCP, ring_ABD, ring_ABD_quat)
    pinky_link1 = calc_first_link_length(pinky_MCP, pinky_ABD, pinky_ABD_quat)

    thumb_length  = np.insert(thumb_length, 0, thumb_link1)
    index_length  = np.insert(index_length, 0, index_link1)
    middle_length  = np.insert(middle_length, 0, middle_link1)
    ring_length  = np.insert(ring_length, 0, ring_link1)
    pinky_length  = np.insert(pinky_length, 0, pinky_link1)

    mpl_fingers_length = np.array([thumb_length, index_length, middle_length, ring_length, pinky_length])

    return mpl_fingers_length


def calc_model_adroit_fingers_length(base_kps):
    adroit_fingers_length = 1.0

    return adroit_fingers_length


# secondly we calc the weight scale
def make_vector(key_point1, key_point2):
    #construct a vector which points key_point2 from key_point1 based on two key points
    dx = key_point2[0] - key_point1[0]
    dy = key_point2[1] - key_point1[1]
    dz = key_point2[2] - key_point1[2]

    vector = np.array([dx, dy, dz])
    
    return vector


def calc_vector_length(vector):
    vector_length = np.sqrt(np.square(vector[0]) + np.square(vector[1]) + np.square(vector[2]))
    
    return vector_length


def calc_finger_length(key_point0, key_point1, key_point2, key_point3, key_point4):
    link1 = calc_vector_length(make_vector(key_point0, key_point1))
    link2 = calc_vector_length(make_vector(key_point1, key_point2))
    link3 = calc_vector_length(make_vector(key_point2, key_point3))
    link4 = calc_vector_length(make_vector(key_point3, key_point4))
    
    finger_length = np.array([link1, link2, link3, link4])
    
    return finger_length


def calc_human_fingers_length(hand_kps):
    # the human fingers length unit:millimeter(mm)
    thumb_length = calc_finger_length(hand_kps[0], hand_kps[17], hand_kps[18], hand_kps[19], hand_kps[20])
    index_length = calc_finger_length(hand_kps[0], hand_kps[1], hand_kps[2], hand_kps[3], hand_kps[4])
    middle_length = calc_finger_length(hand_kps[0], hand_kps[5], hand_kps[6], hand_kps[7], hand_kps[8])
    ring_length = calc_finger_length(hand_kps[0], hand_kps[9], hand_kps[10], hand_kps[11], hand_kps[12])
    pinky_length = calc_finger_length(hand_kps[0], hand_kps[13], hand_kps[14], hand_kps[15], hand_kps[16])
    
    human_fingers_length = np.array([thumb_length, index_length, middle_length, ring_length, pinky_length])
    
    return human_fingers_length


def calc_weight_factors(model_fingers_length, human_fingers_length):
    model_fingers_length = np.array(model_fingers_length).reshape(20,1)
    human_fingers_length = np.array(human_fingers_length).reshape(20,1)
    factors_temp = model_fingers_length / human_fingers_length
    
    weight_factors = np.array(factors_temp).reshape(5,4)
    
    return weight_factors
    

# thirdly we map the human hand keypoints to mechanical hand keypoints
## calc the orientation vector of the finger
def calc_orientation_unit_vector(key_point1, key_point2):
    orientation_vector = make_vector(key_point1, key_point2)
    orientation_vector_length = calc_vector_length(orientation_vector)

    orientation_unit_vector = orientation_vector / orientation_vector_length

    return orientation_unit_vector


def calc_model_finger_keypoints(seq, ABD_pos, finger_kps, finger_weight_factors, human_fingers_length):
    # calc every finger joint keypoints for the model
    MCP = calc_orientation_unit_vector(ABD_pos, finger_kps[0]) * finger_weight_factors[seq][0] * human_fingers_length[seq][0] + finger_kps[0]
    PIP = calc_orientation_unit_vector(finger_kps[0], finger_kps[1]) * finger_weight_factors[seq][1] * human_fingers_length[seq][1] + MCP
    DIP = calc_orientation_unit_vector(finger_kps[1], finger_kps[2]) * finger_weight_factors[seq][2] * human_fingers_length[seq][2] + PIP
    TIP = calc_orientation_unit_vector(finger_kps[2], finger_kps[3]) * finger_weight_factors[seq][3] * human_fingers_length[seq][3] + DIP

    model_finger_keypoints = np.array([MCP, PIP, DIP, TIP])

    return model_finger_keypoints


def calc_model_mpl_keypoints(hand_kps):
    # calc the MPL model link length weight factors
    human_fingers_length = calc_human_fingers_length(hand_kps)
    mpl_fingers_length = calc_model_mpl_fingers_length() * 1000.0 #change the unit from meter to millimeter
    mpl_weight_factors = calc_weight_factors(mpl_fingers_length, human_fingers_length) #(5,4)

    ABD = hand_kps[0]
    hand_kps = np.delete(hand_kps,0,0).reshape(5,4,3)

    thumb_kps = calc_model_finger_keypoints(0, ABD, hand_kps[4], mpl_weight_factors, human_fingers_length)
    index_kps = calc_model_finger_keypoints(1, ABD, hand_kps[0], mpl_weight_factors, human_fingers_length)
    middle_kps = calc_model_finger_keypoints(2, ABD, hand_kps[1], mpl_weight_factors, human_fingers_length)
    ring_kps = calc_model_finger_keypoints(3, ABD, hand_kps[2], mpl_weight_factors, human_fingers_length)
    pinky_kps = calc_model_finger_keypoints(4, ABD, hand_kps[3], mpl_weight_factors, human_fingers_length)

    mpl_kps = np.append(ABD, np.array([index_kps, middle_kps, ring_kps, pinky_kps, thumb_kps]).reshape(20,3)).reshape(21,3)

    return mpl_kps


def calc_model_adroit_keypoints(hand_kps):
    adroit_kps = 0.0
    return adroit_kps

