from joint_position_calculator.get_angle import *
import numpy as np


def get_last_three_finger_joints(key_point0, key_point1, key_point2, key_point3, key_point4):
    #calc last three joints for five fingers(thumb, index, middle, ring, pinky) based on each four key points
    joint1 = calc_vectors_angle(make_vector(key_point0,key_point1), make_vector(key_point1,key_point2))
    joint2 = calc_vectors_angle(make_vector(key_point1,key_point2), make_vector(key_point2,key_point3))
    joint3 = calc_vectors_angle(make_vector(key_point2,key_point3), make_vector(key_point3,key_point4))
    
    last_three_finger_joints = np.array([joint1, joint2, joint3])
    
    return last_three_finger_joints


def check_finger_plane_normal_vector_right(plane_normal_vector, hand_kps):
    # make sure the calculated finger plane normal vector points at the right direction in regular situation
    ref_vector = make_vector(hand_kps[9],hand_kps[1])
    vectors_angle = calc_vectors_angle(ref_vector, plane_normal_vector)
    if vectors_angle <= np.pi/2:
        direction = 1.0
        plane_normal_vector = direction * plane_normal_vector
    elif vectors_angle > np.pi/2:
        direction = -1.0
        plane_normal_vector = direction * plane_normal_vector

    return plane_normal_vector


def check_four_finger_direction(vector_ref, plane_normal_vector, finger_angle, hand_kps):
    # check the first joint rotation direction of four fingers and make sure them right(left is plus/right is minus)
    plane_normal_vector = check_finger_plane_normal_vector_right(plane_normal_vector, hand_kps)
    vectors_angle = calc_vectors_angle(vector_ref, plane_normal_vector)
    if vectors_angle <= np.pi/2:
        direction = 1.0
        finger_angle = direction * finger_angle
    elif vectors_angle > np.pi/2:
        direction = -1.0
        finger_angle = direction * finger_angle
    
    return finger_angle
   

def get_hand_joint_pose(hand_kps):
    #calc the last three joint angles(joint1, joint2, joint3) of five fingers(thumb, index, middle, ring, pinky)
    hand_joint_pose = dict()
    hand_joint_pose.update(thumb = get_last_three_finger_joints(hand_kps[0], hand_kps[17], hand_kps[18], hand_kps[19], hand_kps[20]).tolist())
    hand_joint_pose.update(index = get_last_three_finger_joints(hand_kps[0], hand_kps[1], hand_kps[2], hand_kps[3], hand_kps[4]).tolist())
    hand_joint_pose.update(middle = get_last_three_finger_joints(hand_kps[0], hand_kps[5], hand_kps[6], hand_kps[7], hand_kps[8]).tolist())
    hand_joint_pose.update(ring = get_last_three_finger_joints(hand_kps[0], hand_kps[9], hand_kps[10], hand_kps[11], hand_kps[12]).tolist())
    hand_joint_pose.update(pinky = get_last_three_finger_joints(hand_kps[0], hand_kps[13], hand_kps[14], hand_kps[15], hand_kps[16]).tolist())
    

    #construct the palm/thumb/index/middle/ring/pinky plane based on three key points and calc the normal vector of the palm/thumb/index/middle/ring/pinky plane
#     palm_plane_normal_vector = calc_plane_normal_vector(hand_kps[0], hand_kps[1], hand_kps[9]) #up direction
    palm_plane_normal_vector = calc_plane_normal_vector(hand_kps[0], hand_kps[1], hand_kps[5]) #up direction(for accuracy check)
    thumb_plane_normal_vector = calc_plane_normal_vector(hand_kps[17], hand_kps[18], hand_kps[19])
    index_plane_normal_vector = calc_plane_normal_vector(hand_kps[1], hand_kps[2], hand_kps[3])
    middle_plane_normal_vector = calc_plane_normal_vector(hand_kps[5], hand_kps[6], hand_kps[7])
    ring_plane_normal_vector = calc_plane_normal_vector(hand_kps[9], hand_kps[10], hand_kps[11])
    pinky_plane_normal_vector = calc_plane_normal_vector(hand_kps[13], hand_kps[14], hand_kps[15])


    #calc the first joint0 of the thumb finger
    ##construct the thumb vector for calculation
    thumb_v = make_vector(hand_kps[0], hand_kps[17]) 
    ##calc the angle between thumb_v vector and the palm plane(sometimes need to check the angle whether is over 90 degree)
#     thumb_proj_vector = calc_vector_projection_onto_plane(thumb_v, palm_plane_normal_vector)
#     base_thumb_plane_normal_vector = np.cross(thumb_proj_vector, palm_plane_normal_vector)
    base_thumb_plane_normal_vector = np.array([-0.70710678, -0.5, -0.5])#based on the MPL initial pose plane normal 
    thumb_joint0 = calc_plane_plane_angle(base_thumb_plane_normal_vector, thumb_plane_normal_vector)
    
    hand_joint_pose['thumb'].insert(0, thumb_joint0)


    #calc the first joint0 of rest four fingers(index, middle, ring, pinky)
    ##construct the reference vector and vectors of the four fingers
    ref_vector = make_vector(hand_kps[0],hand_kps[5]) #forward direction
    ##calc the normal vector of the reference plane based on the reference vector and normal vector of the palm plane
    ref_plane_normal_vector = np.cross(ref_vector, palm_plane_normal_vector) #right direction

    ##calc the first joints of the rest four fingers(index, middle, ring, pinky)(needn't check whether the angles is over 90 degree cause they cannot be)
    index_joint0 = calc_plane_plane_angle(ref_plane_normal_vector, index_plane_normal_vector)
    middle_joint0 = calc_plane_plane_angle(ref_plane_normal_vector, middle_plane_normal_vector)
    ring_joint0 = calc_plane_plane_angle(ref_plane_normal_vector, ring_plane_normal_vector)
    pinky_joint0 = calc_plane_plane_angle(ref_plane_normal_vector, pinky_plane_normal_vector)
    
    ##need to check the plus or minus of the calculated angles and determine their directions
    index_joint0 = check_four_finger_direction(ref_vector, index_plane_normal_vector, index_joint0, hand_kps)
    middle_joint0 = check_four_finger_direction(ref_vector, middle_plane_normal_vector, middle_joint0, hand_kps)
    ring_joint0 = check_four_finger_direction(ref_vector, ring_plane_normal_vector, ring_joint0, hand_kps)
    pinky_joint0 = check_four_finger_direction(ref_vector, pinky_plane_normal_vector, pinky_joint0, hand_kps)

    hand_joint_pose['index'].insert(0, index_joint0)
    hand_joint_pose['middle'].insert(0, middle_joint0)
    hand_joint_pose['ring'].insert(0, ring_joint0)
    hand_joint_pose['pinky'].insert(0, pinky_joint0)

    return hand_joint_pose

