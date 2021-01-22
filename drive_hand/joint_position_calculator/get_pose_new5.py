from joint_position_calculator.get_angle_new5 import *
import numpy as np


def get_finger_links(key_point0, key_point1, key_point2, key_point3, key_point4):
    # construct four link vectors based on five finger keypoints(the direction is from the palm points to the fingertip)
    link1 = make_vector(key_point0,key_point1)
    link2 = make_vector(key_point1,key_point2)
    link3 = make_vector(key_point2,key_point3)
    link4 = make_vector(key_point3,key_point4)

    finger_links = np.array([link1, link2, link3, link4])

    return finger_links


def get_hand_links(hand_kps):
    # calc the hand links(20 links) based on 21 hand keypoints and return the hand links dict
    hand_links = dict()
    hand_links.update(thumb = get_finger_links(hand_kps[0], hand_kps[17], hand_kps[18], hand_kps[19], hand_kps[20]).tolist())
    hand_links.update(index = get_finger_links(hand_kps[0], hand_kps[1], hand_kps[2], hand_kps[3], hand_kps[4]).tolist())
    hand_links.update(middle = get_finger_links(hand_kps[0], hand_kps[5], hand_kps[6], hand_kps[7], hand_kps[8]).tolist())
    hand_links.update(ring = get_finger_links(hand_kps[0], hand_kps[9], hand_kps[10], hand_kps[11], hand_kps[12]).tolist())
    hand_links.update(pinky = get_finger_links(hand_kps[0], hand_kps[13], hand_kps[14], hand_kps[15], hand_kps[16]).tolist())

    return hand_links
    

def get_ref_plane_normal_vector(hand_kps):
    # construct the finger plane based on three key points and calc the normal vector of the palm/thumb/index/middle/ring/pinky plane
    palm_pnv = calc_plane_normal_vector_from_points(hand_kps[0], hand_kps[1], hand_kps[5]) #up direction from the palm(for accuracy)
    thumb_pnv = calc_plane_normal_vector_from_points(hand_kps[17], hand_kps[18], hand_kps[19])
    index_pnv = calc_plane_normal_vector_from_points(hand_kps[1], hand_kps[2], hand_kps[3])
    middle_pnv = calc_plane_normal_vector_from_points(hand_kps[5], hand_kps[6], hand_kps[7])
    ring_pnv = calc_plane_normal_vector_from_points(hand_kps[9], hand_kps[10], hand_kps[11])
    pinky_pnv = calc_plane_normal_vector_from_points(hand_kps[13], hand_kps[14], hand_kps[15])
    
    ref_plane_normal_vector = np.array([thumb_pnv, index_pnv, middle_pnv, ring_pnv, pinky_pnv, palm_pnv])
    
    return ref_plane_normal_vector


def check_finger_last_three_joint_direction(vector1, vector2, ref_vector):
    # check each finger joint direction and make them reasonable(if joint is over the range(minus value), then think it stretch straight and set it zero)
    joint_plane_normal_vector = calc_plane_normal_vector_from_vectors(vector1, vector2)
    vectors_angle = calc_vectors_angle(ref_vector, joint_plane_normal_vector)
    if vectors_angle <= np.pi/2:
        direction = 1.0
    elif vectors_angle > np.pi/2:
        direction = -1.0
        #direction = 0.0 #think it stretch straight
        
    return direction


def get_finger_last_three_joints(link1, link2, link3, link4, finger_plane_normal_vector, finger_ref_vector):
    # calc last three joints for five fingers(thumb, index, middle, ring, pinky) based on each four link vectors, finger_plane_normal_vector and finger_ref_vector
    # the direction of finger plane normal vector has no effect on the projection of link1 onto the finger plane
    link1_proj_vector = calc_vector_projection_onto_plane(link1, finger_plane_normal_vector)
    joint1 = calc_vectors_angle(link1_proj_vector, link2) * check_finger_last_three_joint_direction(link1_proj_vector, link2, finger_ref_vector)

    # joint1 = calc_vectors_angle(make_vector(key_point0,key_point1), make_vector(key_point1,key_point2))
    joint2 = calc_vectors_angle(link2, link3) * check_finger_last_three_joint_direction(link2, link3, finger_ref_vector)
    joint3 = calc_vectors_angle(link3, link4) * check_finger_last_three_joint_direction(link3, link4, finger_ref_vector)
    
    finger_last_three_joints = np.array([joint1, joint2, joint3])
    
    return finger_last_three_joints


def check_finger_plane_normal_vector_direction(finger_plane_normal_vector, finger_pnv_direction_ref_vector):
    # check the direction of calculated finger plane normal vector and change it points at the right direction in regular situation if it's not
    vectors_angle = calc_vectors_angle(finger_plane_normal_vector, finger_pnv_direction_ref_vector) #the reference vector points to right direction(for accuracy)
    if vectors_angle <= np.pi/2:
        direction = 1.0
    elif vectors_angle > np.pi/2:
        direction = -1.0

    return direction


def check_finger_first_joint_direction(finger_plane_normal_vector, finger_pnv_direction_ref_vector, finger_pnv_proj_ref_pnv, finger_first_joint_ref_vector):
    # check the first joint rotation direction of five fingers(left is plus/right is minus at the condition that palm_pnv points up direction)
    finger_plane_normal_vector = finger_plane_normal_vector * check_finger_plane_normal_vector_direction(finger_plane_normal_vector, finger_pnv_direction_ref_vector)

    # calc the projection of finger plane normal vector onto palm plane sothat makes the joint0 rotate angle right(compensate the roll of finger plane)
    finger_plane_normal_vector = calc_vector_projection_onto_plane(finger_plane_normal_vector, finger_pnv_proj_ref_pnv)

    vectors_angle = calc_vectors_angle(finger_plane_normal_vector, finger_first_joint_ref_vector)
    if vectors_angle <= np.pi/2:
        direction = 1.0
    elif vectors_angle > np.pi/2:
        direction = -1.0
    
    return direction, finger_plane_normal_vector


def get_finger_first_joint(finger_plane_normal_vector, finger_pnv_direction_ref_vector, finger_pnv_proj_ref_pnv, finger_first_joint_ref_vector, palm_ref_plane_normal_vector):
    # calc first joint of each finger based on the angle between finger plane and reference base plane
    direction, finger_plane_normal_vector = check_finger_first_joint_direction(finger_plane_normal_vector, finger_pnv_direction_ref_vector, finger_pnv_proj_ref_pnv, finger_first_joint_ref_vector)
    finger_first_joint = calc_plane_plane_angle(finger_plane_normal_vector, palm_ref_plane_normal_vector) * direction

    return finger_first_joint


# def get_thumb_first_joint(thumb_plane_normal_vector, thumb_pnv_direction_ref_vector, thumb_ref_plane_normal_vector):
#     # make sure the normal vector of thumb plane and thumb reference plane point the same direction sothat the angle could be over 90 degree
#     thumb_plane_normal_vector = thumb_plane_normal_vector * check_finger_plane_normal_vector_direction(thumb_plane_normal_vector, thumb_pnv_direction_ref_vector) #for the right direction
#     thumb_first_joint = calc_vectors_angle(thumb_plane_normal_vector, thumb_ref_plane_normal_vector)

#     return thumb_first_joint


def get_thumb_first_joint(thumb_plane_normal_vector, thumb_ref_plane_normal_vector):
    # another method for calculating thumb first joint based on the new thumb base reference plane
    # make sure the normal vector of thumb plane and thumb reference plane point the same direction sothat the angle could be over 90 degree
    thumb_first_joint = calc_vectors_angle(thumb_plane_normal_vector, thumb_ref_plane_normal_vector)

    return thumb_first_joint


def get_hand_joint_pose(hand_kps):
    # calc the hand joint pose based on the 21 hand keypoints from annotated/estimated data
    # firstly, calc last three joints of five finger(thumb/index/middle/ring/pinky) based on hand keypoints
    hand_links = get_hand_links(hand_kps)
    ref_plane_normal_vector = get_ref_plane_normal_vector(hand_kps)
    finger_ref_vector_thumb = make_vector(hand_kps[1], hand_kps[17]) #point to right direction at the condition that palm_pnv points up direction(for accuracy)
    finger_ref_vector_index = make_vector(hand_kps[5], hand_kps[1])
    finger_ref_vector_middle = make_vector(hand_kps[9], hand_kps[5])
    finger_ref_vector_ring = make_vector(hand_kps[13], hand_kps[9])
    finger_ref_vector_pinky = make_vector(hand_kps[13], hand_kps[9])

    hand_joint_pose = dict()
    hand_joint_pose.update(thumb = get_finger_last_three_joints(hand_links['thumb'][0], hand_links['thumb'][1], hand_links['thumb'][2], hand_links['thumb'][3], ref_plane_normal_vector[0], finger_ref_vector_thumb).tolist())
    hand_joint_pose.update(index = get_finger_last_three_joints(hand_links['index'][0], hand_links['index'][1], hand_links['index'][2], hand_links['index'][3], ref_plane_normal_vector[1], finger_ref_vector_index).tolist())
    hand_joint_pose.update(middle = get_finger_last_three_joints(hand_links['middle'][0], hand_links['middle'][1], hand_links['middle'][2], hand_links['middle'][3], ref_plane_normal_vector[2], finger_ref_vector_middle).tolist())
    hand_joint_pose.update(ring = get_finger_last_three_joints(hand_links['ring'][0], hand_links['ring'][1], hand_links['ring'][2], hand_links['ring'][3], ref_plane_normal_vector[3], finger_ref_vector_ring).tolist())
    hand_joint_pose.update(pinky = get_finger_last_three_joints(hand_links['pinky'][0], hand_links['pinky'][1], hand_links['pinky'][2], hand_links['pinky'][3], ref_plane_normal_vector[4], finger_ref_vector_pinky).tolist())

    # secondly, calc first joint of five finger based on the angle between finger plane and reference base plane
    finger_pnv_direction_ref_vector_index = make_vector(hand_kps[5], hand_kps[1])
    finger_pnv_direction_ref_vector_middle = make_vector(hand_kps[5], hand_kps[1])
    finger_pnv_direction_ref_vector_ring = make_vector(hand_kps[9], hand_kps[5])
    finger_pnv_direction_ref_vector_pinky = make_vector(hand_kps[13], hand_kps[9])

    finger_pnv_proj_ref_pnv_index = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], hand_kps[1]), make_vector(hand_kps[0], hand_kps[5]))
    finger_pnv_proj_ref_pnv_middle = finger_pnv_proj_ref_pnv_index
    finger_pnv_proj_ref_pnv_ring = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], hand_kps[5]), make_vector(hand_kps[0], hand_kps[9]))
    finger_pnv_proj_ref_pnv_pinky = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], hand_kps[9]), make_vector(hand_kps[0], hand_kps[13]))

    finger_first_joint_ref_vector_index = make_vector(hand_kps[0], hand_kps[1])
    finger_first_joint_ref_vector_middle = make_vector(hand_kps[0], hand_kps[5])
    finger_first_joint_ref_vector_ring = make_vector(hand_kps[0], hand_kps[9])
    finger_first_joint_ref_vector_pinky = make_vector(hand_kps[0], hand_kps[13])

    # palm_ref_plane_normal_vector = np.cross(finger_first_joint_ref_vector, ref_plane_normal_vector[5]) #points to right direction at the condition that palm_pnv points up direction(for accuracy)
    index_ref_plane_normal_vector = np.cross(finger_first_joint_ref_vector_index, finger_pnv_proj_ref_pnv_index) #points to right direction at the condition that palm_pnv points up direction(for accuracy)
    middle_ref_plane_normal_vector = np.cross(finger_first_joint_ref_vector_middle, finger_pnv_proj_ref_pnv_middle) #points to right direction at the condition that palm_pnv points up direction(for accuracy)
    ring_ref_plane_normal_vector = np.cross(finger_first_joint_ref_vector_ring, finger_pnv_proj_ref_pnv_ring) #points to right direction at the condition that palm_pnv points up direction(for accuracy)
    pinky_ref_plane_normal_vector = np.cross(finger_first_joint_ref_vector_pinky, finger_pnv_proj_ref_pnv_pinky) #points to right direction at the condition that palm_pnv points up direction(for accuracy)

    # calc the rotate angle of every finger plane relative to the each finger
    # index_joint0 = get_finger_first_joint(ref_plane_normal_vector[1], finger_pnv_direction_ref_vector_index, finger_pnv_proj_ref_pnv_index, finger_first_joint_ref_vector_index, index_ref_plane_normal_vector)
    # middle_joint0 = get_finger_first_joint(ref_plane_normal_vector[2], finger_pnv_direction_ref_vector_middle, finger_pnv_proj_ref_pnv_middle, finger_first_joint_ref_vector_middle, middle_ref_plane_normal_vector)
    # ring_joint0 = get_finger_first_joint(ref_plane_normal_vector[3], finger_pnv_direction_ref_vector_ring, finger_pnv_proj_ref_pnv_ring, finger_first_joint_ref_vector_ring, ring_ref_plane_normal_vector)
    # pinky_joint0 = get_finger_first_joint(ref_plane_normal_vector[4], finger_pnv_direction_ref_vector_pinky, finger_pnv_proj_ref_pnv_pinky, finger_first_joint_ref_vector_pinky, pinky_ref_plane_normal_vector)
    index_plane_normal_vector = calc_plane_normal_vector_from_vectors(finger_first_joint_ref_vector_index, make_vector(hand_kps[1], hand_kps[2]))
    middle_plane_normal_vector = calc_plane_normal_vector_from_vectors(finger_first_joint_ref_vector_middle, make_vector(hand_kps[5], hand_kps[6]))
    ring_plane_normal_vector = calc_plane_normal_vector_from_vectors(finger_first_joint_ref_vector_ring, make_vector(hand_kps[9], hand_kps[10]))
    pinky_plane_normal_vector = calc_plane_normal_vector_from_vectors(finger_first_joint_ref_vector_pinky, make_vector(hand_kps[13], hand_kps[14]))
    index_joint0 = get_finger_first_joint(index_plane_normal_vector, finger_pnv_direction_ref_vector_index, finger_pnv_proj_ref_pnv_index, finger_first_joint_ref_vector_index, index_ref_plane_normal_vector)
    middle_joint0 = get_finger_first_joint(middle_plane_normal_vector, finger_pnv_direction_ref_vector_middle, finger_pnv_proj_ref_pnv_middle, finger_first_joint_ref_vector_middle, middle_ref_plane_normal_vector)
    ring_joint0 = get_finger_first_joint(ring_plane_normal_vector, finger_pnv_direction_ref_vector_ring, finger_pnv_proj_ref_pnv_ring, finger_first_joint_ref_vector_ring, ring_ref_plane_normal_vector)
    pinky_joint0 = get_finger_first_joint(pinky_plane_normal_vector, finger_pnv_direction_ref_vector_pinky, finger_pnv_proj_ref_pnv_pinky, finger_first_joint_ref_vector_pinky, pinky_ref_plane_normal_vector)
    
    # construct the projection of finger calculation plane onto the zero base plane and calc the rotate angle of the zero base plane(middle finger:middle_ref_plane_normal)
    # index_ref_plane_normal_vector_proj = index_ref_plane_normal_vector
    # middle_ref_plane_normal_vector_proj = middle_ref_plane_normal_vector
    index_ref_plane_normal_vector_proj = calc_vector_projection_onto_plane(index_ref_plane_normal_vector, finger_pnv_proj_ref_pnv_middle)
    middle_ref_plane_normal_vector_proj = calc_vector_projection_onto_plane(middle_ref_plane_normal_vector, finger_pnv_proj_ref_pnv_middle)
    ring_ref_plane_normal_vector_proj = calc_vector_projection_onto_plane(ring_ref_plane_normal_vector, finger_pnv_proj_ref_pnv_middle)
    pinky_ref_plane_normal_vector_proj = calc_vector_projection_onto_plane(pinky_ref_plane_normal_vector, finger_pnv_proj_ref_pnv_middle)

    # maybe we should choose the plane based on keypoints 0,1,9(now is keypoints 0,1,5)
    # zero_base_pnv = np.cross(make_vector(hand_kps[0], hand_kps[5]), calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], hand_kps[1]), make_vector(hand_kps[0], hand_kps[9])))
    zero_base_pnv = middle_ref_plane_normal_vector
    index_delta = calc_plane_plane_angle(index_ref_plane_normal_vector_proj, zero_base_pnv)
    # middle_delta = 0.0
    middle_delta = calc_plane_plane_angle(index_ref_plane_normal_vector_proj, zero_base_pnv)
    ring_delta = calc_plane_plane_angle(ring_ref_plane_normal_vector_proj, zero_base_pnv)
    pinky_delta = calc_plane_plane_angle(pinky_ref_plane_normal_vector_proj, zero_base_pnv)

    index_joint0 = -index_delta + index_joint0
    middle_joint0 = middle_delta + middle_joint0
    ring_joint0 = ring_delta + ring_joint0
    pinky_joint0 = pinky_delta + pinky_joint0

    hand_joint_pose['index'].insert(0, index_joint0)
    hand_joint_pose['middle'].insert(0, middle_joint0)
    hand_joint_pose['ring'].insert(0, ring_joint0)
    hand_joint_pose['pinky'].insert(0, pinky_joint0)

    # calc the first joint of thumb finger based on the angle between finger plane and reference base plane
    # thumb_ref_plane_normal_vector = np.array([0.5, 0.5, -0.70710678]) #based on the MPL initial pose plane normal(right direction)/rotate 45 degree along x axis
    # thumb_ref_plane_normal_vector = np.array([0.61237244, 0.61237244, -0.5]) #based on the MPL initial pose plane normal(right direction))/rotate 60 degree along x axis
    # thumb_pnv_direction_ref_vector = make_vector(hand_kps[1], hand_kps[17])
    thumb_ref_plane_normal_vector = np.array([0.0, 0.0, -1.0])
    thumb_plane_normal_vector = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], hand_kps[17]), make_vector(hand_kps[0], hand_kps[1]))
    
    # thumb_joint0 = calc_plane_plane_angle(thumb_ref_plane_normal_vector, ref_plane_normal_vector[0]) 
    # thumb_joint0 = get_thumb_first_joint(ref_plane_normal_vector[0], thumb_pnv_direction_ref_vector, thumb_ref_plane_normal_vector)
    thumb_joint0 = get_thumb_first_joint(thumb_plane_normal_vector, thumb_ref_plane_normal_vector)
    hand_joint_pose['thumb'].insert(0, thumb_joint0)

    return hand_joint_pose

