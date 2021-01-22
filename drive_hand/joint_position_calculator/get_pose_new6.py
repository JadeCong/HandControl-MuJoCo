"""
Calculate the 20 joint angles of human hand for mapping joints of mechanical hand in mujoco and controling it.
Statement: for abbreviation
          pnv = plane_normal_vector
          drv = direction_ref_vector
"""
import numpy as np
import pyquaternion as pq
from joint_position_calculator.get_angle_new6 import *


"""
construct hand_links as vectors and ref_plane_normal_vectors and local_finger_base_plane_normal_vectors for calculation of finger joint angles
"""
def euler_rotate_zyx(alpha, beta, gamma):
    # euler rotate in 3d cartesian space for vector(rotate angle:x:alpha/y:beta/z:gamma)
    # (zyx is regular euler rotate sequence in mujoco scene, but fixed axis rotate sequence is xyz)
    Rx = np.matrix([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    # the euler rotate axis sequence: z(gamma), y(beta), x(alpha)
    transform = Rz * Ry * Rx

    return transform


def euler_rotate_xyz(alpha, beta, gamma):
    # euler rotate in 3d cartesian space for vector(rotate angle:x:alpha/y:beta/z:gamma)
    # (xyz is regular euler rotate sequence in mujoco scene, but fixed axis rotate sequence is zyx)
    Rx = np.matrix([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    # the euler rotate axis sequence: x(alpha), y(beta), z(gamma)
    transform = Rx * Ry * Rz

    return transform


def euler_rotate_yzx(alpha, beta, gamma):
    # euler rotate in 3d cartesian space for vector(rotate angle:x:alpha/y:beta/z:gamma)
    # (yzx is regular euler rotate sequence in mujoco scene, but fixed axis rotate sequence is xzy)
    Rx = np.matrix([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    # the euler rotate axis sequence: y(beta), z(gamma), x(alpha)
    transform = Ry * Rz * Rx

    return transform


def euler_rotate_zyz(alpha, beta, gamma):
    # euler rotate in 3d cartesian space for vector(rotate angle:z1:alpha/y:beta/z2:gamma)
    # (z1yz2 is regular euler rotate sequence in mujoco scene, but fixed axis rotate sequence is z2yz1)
    Rz1 = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz2 = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    # the euler rotate axis sequence: z1(alpha), y(beta), z2(gama)
    transform = Rz1 * Ry * Rz2

    return transform


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
    

def get_ref_plane_normal_vectors(hand_kps):
    # construct the finger plane based on three key points(middle three kps of each finger) and calc the normal vector of the thumb/index/middle/ring/pinky/palm plane
    #TODO: caution:sometimes the finger plane normal vector may point to other direction instead of left/right direction
    thumb_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[17], hand_kps[18]), make_vector(hand_kps[18], hand_kps[19]))
    index_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[1], hand_kps[2]), make_vector(hand_kps[2], hand_kps[3]))
    middle_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[5], hand_kps[6]), make_vector(hand_kps[6], hand_kps[7]))
    ring_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[9], hand_kps[10]), make_vector(hand_kps[10], hand_kps[11]))
    pinky_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[13], hand_kps[14]), make_vector(hand_kps[14], hand_kps[15]))

    #TODO:(Done by Jade) construct the palm plane based on five keypoints(0/1/5/9/13), we calculate the middle point between 1 and 5 as palm_kps_right, then the middle point between 9 and 13 as palm_kps_left
    palm_kps_right = (hand_kps[1] + hand_kps[5]) / 2.0
    palm_kps_left = (hand_kps[9] + hand_kps[13]) / 2.0
    palm_pnv = calc_plane_normal_vector_from_vectors(make_vector(hand_kps[0], palm_kps_right), make_vector(hand_kps[0], palm_kps_left)) # make sure the palm_pnv points up direction
    
    ref_plane_normal_vectors = np.array([thumb_pnv, index_pnv, middle_pnv, ring_pnv, pinky_pnv, palm_pnv])
    
    return ref_plane_normal_vectors


def get_finger_pnv_direction_ref_vectors(hand_kps):
    # construct the direction reference vector of five finger pnvs for calculating last three joints of each finger
    thumb_pnv_drv = make_vector(hand_kps[1], hand_kps[17])
    index_pnv_drv = make_vector(hand_kps[5], hand_kps[1])
    middle_pnv_drv = make_vector(hand_kps[5], hand_kps[1])
    ring_pnv_drv = make_vector(hand_kps[9], hand_kps[5])
    pinky_pnv_drv = make_vector(hand_kps[13], hand_kps[9])

    finger_pnv_direction_ref_vectors = np.array([thumb_pnv_drv, index_pnv_drv, middle_pnv_drv, ring_pnv_drv, pinky_pnv_drv])

    return finger_pnv_direction_ref_vectors


def get_finger_first_joint_direction_ref_vectors(hand_kps):
    # construct the direction reference vector of first joint for each finger
    thumb_fj_drv = make_vector(hand_kps[0], hand_kps[17])
    index_fj_drv = make_vector(hand_kps[0], hand_kps[1])
    middle_fj_drv = make_vector(hand_kps[0], hand_kps[5])
    ring_fj_drv = make_vector(hand_kps[0], hand_kps[9])
    pinky_fj_drv = make_vector(hand_kps[0], hand_kps[13])

    finger_first_joint_drvs = np.array([thumb_fj_drv, index_fj_drv, middle_fj_drv, ring_fj_drv, pinky_fj_drv])

    return finger_first_joint_drvs


def get_local_finger_base_plane_normal_vector(vector1, vector2, vector_in_plane):
    # construct the local_finger_base_plane_normal_vector for calculating finger_first_joint in the function get_finger_first_joint_plus
    #TODO:(Done by Jade) make sure that the local_finger_base_pnv points to right direction and the local_finger_pnv_proj_ref_pnv points to up direction
    local_finger_pnv_proj_ref_pnv = calc_plane_normal_vector_from_vectors(vector1, vector2) # now the local_finger_pnv_proj_ref_pnv points to up direction
    local_finger_base_pnv = np.cross(vector_in_plane, local_finger_pnv_proj_ref_pnv) # now the local_finger_base_pnv points to right direction

    return local_finger_pnv_proj_ref_pnv, local_finger_base_pnv


def get_local_finger_pnv_proj_ref_pnvs_and_local_finger_base_pnvs(hand_kps):
    # get the local_finger_pnv_proj_ref_pnvs and local_finger_base_pnvs of four fingers(index/middle/ring/pinky) for calculating the first joints
    hand_links = get_hand_links(hand_kps)
    index_pnv_proj_ref_pnv, index_base_pnv = get_local_finger_base_plane_normal_vector(hand_links['index'][0], hand_links['middle'][0], hand_links['index'][0])
    middle_pnv_proj_ref_pnv, middle_base_pnv = get_local_finger_base_plane_normal_vector(hand_links['index'][0], hand_links['middle'][0], hand_links['middle'][0])
    ring_pnv_proj_ref_pnv, ring_base_pnv = get_local_finger_base_plane_normal_vector(hand_links['middle'][0], hand_links['ring'][0], hand_links['ring'][0])
    pinky_pnv_proj_ref_pnv, pinky_base_pnv = get_local_finger_base_plane_normal_vector(hand_links['ring'][0], hand_links['pinky'][0], hand_links['pinky'][0])

    local_finger_pnv_proj_ref_pnvs = np.array([index_pnv_proj_ref_pnv, middle_pnv_proj_ref_pnv, ring_pnv_proj_ref_pnv, pinky_pnv_proj_ref_pnv])
    local_finger_base_pnvs = np.array([index_base_pnv, middle_base_pnv, ring_base_pnv, pinky_base_pnv])

    return local_finger_pnv_proj_ref_pnvs, local_finger_base_pnvs


"""
calculate the last three joints of five fingers(thumb/index/middle/ring/pinky) based on the angle between link projections
"""
def check_finger_plane_normal_vector_direction(finger_pnv, finger_pnv_drv):
    # check the direction of calculated finger_pnv and change it points at the right direction in regular situation if it's not
    #TODO:(Done by Jade) make sure that the reference vector points to right direction(for accuracy)
    vectors_angle = calc_vectors_angle(finger_pnv, finger_pnv_drv)
    ##check the vectors_angle matches the requirement or not
    assert (0.0 <= vectors_angle and vectors_angle <= np.pi), "Wrong vectors_angle for finger_pnv!"
    direction = 0.0
    if vectors_angle <= np.pi/2.0:
        direction = 1.0
    elif vectors_angle > np.pi/2.0:
        direction = -1.0

    return direction


def check_finger_last_three_joint_direction(vector1, vector2, direction_ref_vector):
    # check each finger last three joint direction and make them reasonable(if joint is over the range(minus value), then think it stretch straight and set it zero)
    joint_plane_normal_vector = calc_plane_normal_vector_from_vectors(vector1, vector2) # hold vector1 to vector2(the joint_pnv direction don't affect the result)
    vectors_angle = calc_vectors_angle(direction_ref_vector, joint_plane_normal_vector)
    ##check the vectors_angle matches the requirement or not
    assert (0.0 <= vectors_angle and vectors_angle <= np.pi), "Wrong vectors_angle for joint_pnv!"
    direction = 0.0
    if vectors_angle <= np.pi/2.0:
        direction = 1.0
    elif vectors_angle > np.pi/2.0:
        direction = -1.0
        #direction = 0.0 #think it stretch straight
        
    return direction


def get_finger_last_three_joints(link1, link2, link3, link4, finger_pnv, finger_pnv_drv):
    # calc last three joints for five fingers based on each four link vectors, finger_pnv and finger_pnv_drv
    # the direction of finger_pnv has no effect on the projection of links onto the finger plane(but also make sure the finger_pnv points to right direction)
    direction = check_finger_plane_normal_vector_direction(finger_pnv, finger_pnv_drv)
    finger_pnv = finger_pnv * direction

    link1_proj_vector = calc_vector_projection_onto_plane(link1, finger_pnv)
    link2_proj_vector = calc_vector_projection_onto_plane(link2, finger_pnv)
    link3_proj_vector = calc_vector_projection_onto_plane(link3, finger_pnv)
    link4_proj_vector = calc_vector_projection_onto_plane(link4, finger_pnv)
    #TODO:(Done by Jade) make sure that finger_pnv_drv points to right direction and is vertical to the finger plane(at the condition that finger_pnv must point to right direction)
    finger_drv_proj_vector = calc_vector_projection_onto_vector(finger_pnv_drv, finger_pnv) 

    direction1 = check_finger_last_three_joint_direction(link1_proj_vector, link2_proj_vector, finger_drv_proj_vector)
    direction2 = check_finger_last_three_joint_direction(link2_proj_vector, link3_proj_vector, finger_drv_proj_vector)
    direction3 = check_finger_last_three_joint_direction(link3_proj_vector, link4_proj_vector, finger_drv_proj_vector)
    joint1 = calc_vectors_angle(link1_proj_vector, link2_proj_vector) * direction1
    joint2 = calc_vectors_angle(link2_proj_vector, link3_proj_vector) * direction2
    joint3 = calc_vectors_angle(link3_proj_vector, link4_proj_vector) * direction3
    
    finger_last_three_joints = np.array([joint1, joint2, joint3])
    
    return finger_last_three_joints


"""
calculate the first joint of four fingers(index/middle/ring/pinky) based on the angle between finger plane and zero base plane
"""
def check_finger_first_joint_direction(finger_pnv, finger_pnv_proj_ref_pnv, finger_first_joint_drv):
    # check the first joint rotation direction of four fingers(left is plus/right is minus at the condition that finger_pnv_proj_ref_pnv points up direction)
    # calc the projection of finger_pnv onto palm plane(finger_pnv_proj_ref_pnv) sothat makes the joint0 compensate the roll error of finger plane
    #TODO:(Done by Jade) make sure that the finger_pnv points to right direction and finger_pnv_proj_ref_pnv points to up direction
    finger_pnv_proj_vector = calc_vector_projection_onto_plane(finger_pnv, finger_pnv_proj_ref_pnv)
    #TODO:(Done by Jade) make sure that finger_first_joint_drv(vector based on kps[0/5]) project onto the finger_pnv_proj_ref_pnv
    finger_first_joint_drv_proj_vector = calc_vector_projection_onto_plane(finger_first_joint_drv, finger_pnv_proj_ref_pnv)

    vectors_angle = calc_vectors_angle(finger_pnv_proj_vector, finger_first_joint_drv_proj_vector)
    assert (0.0 <= vectors_angle and vectors_angle <= np.pi), "Wrong vectors_angle for finger first joint direction!"
    direction = 0.0
    if vectors_angle <= np.pi/2.0:
        direction = 1.0
    elif vectors_angle > np.pi/2.0:
        direction = -1.0
    
    return direction, finger_pnv_proj_vector


# method 1:directly calculate the angle between the finger plane and zero base plane
def get_finger_first_joint(finger_pnv, finger_pnv_drv, finger_pnv_proj_ref_pnv, finger_first_joint_drv, zero_base_pnv):
    # calc first joint of each finger based on the angle between finger plane and zero base plane
    #TODO:(Done by jade) make sure that the zero_base_pnv plane is vertical to palm plane(finger_pnv_proj_ref_pnv plane) and the zero_base_pnv points to right direction
    direction = check_finger_plane_normal_vector_direction(finger_pnv, finger_pnv_drv)
    finger_pnv = finger_pnv * direction

    direction1, finger_pnv_proj_vector = check_finger_first_joint_direction(finger_pnv, finger_pnv_proj_ref_pnv, finger_first_joint_drv)
    finger_first_joint = calc_vectors_angle(finger_pnv_proj_vector, zero_base_pnv) * direction1

    return finger_first_joint


# method 2:calculate the angle between the finger plane and finger ref plane and the angle between finger ref plane and zero base plane, then plus them for joint0
def get_finger_first_joint_plus(finger_pnv, finger_pnv_drv, local_finger_pnv_proj_ref_pnv, local_finger_first_joint_drv, local_finger_base_pnv, zero_base_pnv):
    # calc first joint of each finger based on the angle between finger plane and zero base plane through plus method and direction check
    #TODO:(Done by Jade) make sure that the zero_base_pnv plane is vertical to palm plane and the zero_base_pnv points to right direction
    #TODO:(Done by Jade) and make sure that local_finger_base_pnv points to right direction but not sure local_finger_base_pnv plane is vertical to palm plane
    direction = check_finger_plane_normal_vector_direction(finger_pnv, finger_pnv_drv)
    finger_pnv = finger_pnv * direction

    direction1, local_finger_pnv_proj_vector = check_finger_first_joint_direction(finger_pnv, local_finger_pnv_proj_ref_pnv, local_finger_first_joint_drv)
    joint_delta1 = calc_vectors_angle(local_finger_pnv_proj_vector, local_finger_base_pnv) * direction1

    #TODO:(Done by Jade) make sure that the direction of angle between local_finger_base_pnv plane and zero_base_pnv plane is correct
    zero_base_pnv_proj_vector = calc_vector_projection_onto_plane(zero_base_pnv, local_finger_pnv_proj_ref_pnv)
    # check the direction of joint_delta1 to make it correct
    cross_vector = np.cross(zero_base_pnv_proj_vector, local_finger_base_pnv)
    vectors_angle = calc_vectors_angle(cross_vector, local_finger_pnv_proj_ref_pnv)
    direction2 = 0.0
    if vectors_angle <= np.pi/2.0: #vectors_angle == 0.0
        direction2 = 1.0
    elif vectors_angle > np.pi/2.0: #vector_angle == np.pi
        direction2 = -1.0
    joint_delta2 = calc_vectors_angle(zero_base_pnv_proj_vector, local_finger_base_pnv) * direction2

    finger_first_joint = joint_delta1 + joint_delta2

    return finger_first_joint


"""
calculate the first joint of thumb finger based on the angle between thumb finger plane and thumb base plane
"""
# method 1:directly calculate the angle between thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoints(0,1,17))
def get_thumb_first_joint_simple1(thumb_pnv, thumb_base_pnv, thumb_joint_drv):
    # calc the first joint of thumb finger(construct thumb_pnv plane from three keypoints(0,1,17))
    #TODO:(Done by Jade) make sure that the thumb_pnv and thumb_base_pnv point to up direction
    direction = check_finger_plane_normal_vector_direction(np.cross(thumb_pnv, thumb_base_pnv), thumb_joint_drv)
    thumb_first_joint = calc_vectors_angle(thumb_pnv, thumb_base_pnv) * direction
    
    return thumb_first_joint


# method 2:directly calculate the angle between the projection of thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoints(0,1,17))
def get_thumb_first_joint_simple2(thumb_pnv, thumb_pnv_proj_ref_pnv, thumb_base_pnv):
    # calc the first joint of thumb finger(construct thumb_pnv plane from three keypoints(0,1,17))
    #TODO:(Done by Jade) make sure that the thumb_pnv plane is correct and same as the up direction(at the condition that thumb_base_pnv and thumb_pnv_drv plane points to up direction)
    thumb_pnv_proj_vector = calc_vector_projection_onto_plane(thumb_pnv, thumb_pnv_proj_ref_pnv) # rotate along the axis vector(link_05)
    direction = check_finger_plane_normal_vector_direction(np.cross(thumb_pnv_proj_vector, thumb_base_pnv), thumb_pnv_proj_ref_pnv)
    thumb_first_joint = calc_vectors_angle(thumb_pnv_proj_vector, thumb_base_pnv) * direction
    
    return thumb_first_joint


# method 3:directly calculate the angle between thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(17,18,19))
def get_thumb_first_joint1(thumb_pnv, thumb_pnv_drv, thumb_base_pnv, thumb_joint_drv):
    # calc the first joint of thumb finger(construct thumb_pnv plane from three keypoint(17,18,19))
    #TODO:(Done by Jade) make sure that the thumb_pnv plane is correct and same as the right direction(at the condition that thumb_base_pnv and thumb_pnv_drv points to right direction)
    direction = check_finger_plane_normal_vector_direction(thumb_pnv, thumb_pnv_drv)
    thumb_pnv = thumb_pnv * direction

    direction1 = check_finger_plane_normal_vector_direction(np.cross(thumb_pnv, thumb_base_pnv), thumb_joint_drv)
    thumb_first_joint = calc_vectors_angle(thumb_pnv, thumb_base_pnv) * direction1
    
    return thumb_first_joint


# method 4:directly calculate the angle between the projection of thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(17,18,19))
def get_thumb_first_joint2(thumb_pnv, thumb_pnv_drv, thumb_base_pnv, thumb_pnv_proj_ref_pnv, thumb_joint_drv):
    # calc the first joint of thumb finger(construct thumb_pnv plane from three keypoint(17,18,19))
    #TODO:(Done by Jade) make sure that the thumb_pnv plane is correct and same as the right direction(at the condition that thumb_base_pnv and thumb_pnv_drv points to right direction)
    direction = check_finger_plane_normal_vector_direction(thumb_pnv, thumb_pnv_drv)
    thumb_pnv = thumb_pnv * direction

    thumb_pnv_proj_vector = calc_vector_projection_onto_plane(thumb_pnv, thumb_pnv_proj_ref_pnv)
    thumb_base_pnv_proj_vector = calc_vector_projection_onto_plane(thumb_base_pnv, thumb_pnv_proj_ref_pnv)

    direction1 = check_finger_plane_normal_vector_direction(np.cross(thumb_pnv_proj_vector, thumb_base_pnv_proj_vector), thumb_joint_drv)
    thumb_first_joint = calc_vectors_angle(thumb_pnv_proj_vector, thumb_base_pnv_proj_vector) * direction1
    
    return thumb_first_joint


"""
calculate the 20 finger joint angles of the whole hand
"""
def get_hand_joint_pose(hand_kps):
    # calc the hand joint pose based on 21 hand keypoints from annotated/estimated data
    '''
    firstly, calc last three joints of five fingers(thumb/index/middle/ring/pinky) based on hand keypoints
    '''
    # get all finger links of the hand
    hand_links = get_hand_links(hand_kps)
    ref_pnvs = get_ref_plane_normal_vectors(hand_kps)
    finger_pnv_drvs = get_finger_pnv_direction_ref_vectors(hand_kps) #both point to right direction for accuracy
    
    hand_joint_pose = dict()
    hand_joint_pose.update(thumb = get_finger_last_three_joints(hand_links['thumb'][0], hand_links['thumb'][1], hand_links['thumb'][2], hand_links['thumb'][3], ref_pnvs[0], finger_pnv_drvs[0]).tolist())
    hand_joint_pose.update(index = get_finger_last_three_joints(hand_links['index'][0], hand_links['index'][1], hand_links['index'][2], hand_links['index'][3], ref_pnvs[1], finger_pnv_drvs[1]).tolist())
    hand_joint_pose.update(middle = get_finger_last_three_joints(hand_links['middle'][0], hand_links['middle'][1], hand_links['middle'][2], hand_links['middle'][3], ref_pnvs[2], finger_pnv_drvs[2]).tolist())
    hand_joint_pose.update(ring = get_finger_last_three_joints(hand_links['ring'][0], hand_links['ring'][1], hand_links['ring'][2], hand_links['ring'][3], ref_pnvs[3], finger_pnv_drvs[3]).tolist())
    hand_joint_pose.update(pinky = get_finger_last_three_joints(hand_links['pinky'][0], hand_links['pinky'][1], hand_links['pinky'][2], hand_links['pinky'][3], ref_pnvs[4], finger_pnv_drvs[4]).tolist())

    '''
    secondly, first joint of four fingers(index/middle/ring/pinky) based on the angle between finger plane and zero base plane
    '''
    # method 1:directly calculate the angle between the finger plane and zero base plane
    finger_fj_drvs = get_finger_first_joint_direction_ref_vectors(hand_kps)
    # construct the zero base plane that points to right direction and it must be vertical to the palm plane and the vector(link05) is in it
    zero_base_pnv = np.cross(hand_links['middle'][0], ref_pnvs[5])

    index_joint0 = get_finger_first_joint(ref_pnvs[1], finger_pnv_drvs[1], ref_pnvs[5], finger_fj_drvs[1], zero_base_pnv)
    middle_joint0 = get_finger_first_joint(ref_pnvs[2], finger_pnv_drvs[2], ref_pnvs[5], finger_fj_drvs[2], zero_base_pnv)
    ring_joint0 = get_finger_first_joint(ref_pnvs[3], finger_pnv_drvs[3], ref_pnvs[5], finger_fj_drvs[3], zero_base_pnv)
    pinky_joint0 = get_finger_first_joint(ref_pnvs[4], finger_pnv_drvs[4], ref_pnvs[5], finger_fj_drvs[4], zero_base_pnv)

    hand_joint_pose['index'].insert(0, index_joint0)
    hand_joint_pose['middle'].insert(0, middle_joint0)
    hand_joint_pose['ring'].insert(0, ring_joint0)
    hand_joint_pose['pinky'].insert(0, pinky_joint0)


    # method 2:calculate the angle between the finger plane and finger ref plane and the angle between finger ref plane and zero base plane, then plus them for joint0
    # calc the local_finger_pnv_proj_ref_pnv and local_finger_base_pnv for four fingers
    zero_base_pnv = np.cross(hand_links['middle'][0], ref_pnvs[5])
    local_finger_pnv_proj_ref_pnvs, local_finger_base_pnvs = get_local_finger_pnv_proj_ref_pnvs_and_local_finger_base_pnvs(hand_kps)

    index_joint0 = get_finger_first_joint_plus(ref_pnvs[1], finger_pnv_drvs[1], local_finger_pnv_proj_ref_pnvs[0], hand_links['index'][0], local_finger_base_pnvs[0], zero_base_pnv)
    middle_joint0 = get_finger_first_joint_plus(ref_pnvs[2], finger_pnv_drvs[2], local_finger_pnv_proj_ref_pnvs[1], hand_links['middle'][0], local_finger_base_pnvs[1], zero_base_pnv)
    ring_joint0 = get_finger_first_joint_plus(ref_pnvs[3], finger_pnv_drvs[3], local_finger_pnv_proj_ref_pnvs[2], hand_links['ring'][0], local_finger_base_pnvs[2], zero_base_pnv)
    pinky_joint0 = get_finger_first_joint_plus(ref_pnvs[4], finger_pnv_drvs[4], local_finger_pnv_proj_ref_pnvs[3], hand_links['pinky'][0], local_finger_base_pnvs[3], zero_base_pnv)

    hand_joint_pose['index'].insert(0, index_joint0)
    hand_joint_pose['middle'].insert(0, middle_joint0)
    hand_joint_pose['ring'].insert(0, ring_joint0)
    hand_joint_pose['pinky'].insert(0, pinky_joint0)

    '''
    finally, calc the first joint of thumb finger based on the angle between thumb finger plane and thumb base plane
    '''
    # method 1:directly calculate the angle between thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(0,1,17))
    thumb_base_pnv = np.cross(hand_links['index'][0], hand_links['middle'][0])
    thumb_pnv = np.cross(hand_links['thumb'][0], hand_links['index'][0])
    thumb_joint_drv = hand_links['index'][0]
    thumb_joint0 = get_thumb_first_joint_simple1(thumb_pnv, thumb_base_pnv, thumb_joint_drv)

    hand_joint_pose['thumb'].insert(0, thumb_joint0)


    # method 2:directly calculate the angle between the projection of thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(0,1,17))
    thumb_base_pnv = np.cross(hand_links['index'][0], hand_links['middle'][0])
    thumb_pnv = np.cross(hand_links['thumb'][0], hand_links['index'][0])
    thumb_pnv_proj_ref_pnv = hand_links['middle'][0]
    thumb_joint0 = get_thumb_first_joint_simple2(thumb_pnv, thumb_pnv_proj_ref_pnv, thumb_base_pnv)

    hand_joint_pose['thumb'].insert(0, thumb_joint0)


    # method 3:directly calculate the angle between thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(17,18,19))
    z_axis = -ref_pnvs[5] #up direction
    y_axis = -calc_vector_projection_onto_plane(hand_links['middle'][0], z_axis) #backward direction
    x_axis = np.cross(y_axis, z_axis) #left direction

    x_axis_rot = pq.Quaternion(axis=z_axis, angle=-np.pi/4.0).rotate(x_axis)
    thumb_base_pnv = pq.Quaternion(axis=x_axis_rot, angle=np.pi/4.0).rotate(-z_axis) #calc the thumb_base_pnv(palm points to down direction)

    thumb_pnv = ref_pnvs[0]
    thumb_pnv_drv = finger_pnv_drvs[0]
    thumb_joint_drv = hand_links['middle'][0] #need to check whether the thumb_joint_drv is right for calculation
    thumb_joint0 = get_thumb_first_joint1(thumb_pnv, thumb_pnv_drv, thumb_base_pnv, thumb_joint_drv)

    hand_joint_pose['thumb'].insert(0, thumb_joint0)


    # method 4:directly calculate the angle between the projection of thumb_pnv and thumb_base_pnv(construct thumb_pnv plane from three keypoint(17,18,19))
    z_axis = -ref_pnvs[5] #up direction
    y_axis = -calc_vector_projection_onto_plane(hand_links['middle'][0], z_axis) #backward direction
    x_axis = np.cross(y_axis, z_axis) #left direction

    x_axis_rot = pq.Quaternion(axis=z_axis, angle=-np.pi/4.0).rotate(x_axis)
    thumb_base_pnv = pq.Quaternion(axis=x_axis_rot, angle=np.pi/4.0).rotate(-z_axis) #calc the thumb_base_pnv(palm points to down direction)

    thumb_pnv = ref_pnvs[0]
    thumb_pnv_drv = finger_pnv_drvs[0]
    thumb_pnv_proj_ref_pnv = -y_axis
    thumb_joint_drv = hand_links['middle'][0]
    thumb_joint0 = get_thumb_first_joint2(thumb_pnv, thumb_pnv_drv, thumb_base_pnv, thumb_pnv_proj_ref_pnv, thumb_joint_drv)

    hand_joint_pose['thumb'].insert(0, thumb_joint0)

    return hand_joint_pose

