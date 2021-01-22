import numpy as np 


def make_vector(key_point1, key_point2):
    #construct a vector which points key_point2 from key_point1 based on two key points
    dx = key_point2[0] - key_point1[0]
    dy = key_point2[1] - key_point1[1]
    dz = key_point2[2] - key_point1[2]

    vector = np.zeros(3)
    vector[0] = dx
    vector[1] = dy
    vector[2] = dz
    
    return vector


def calc_vectors_angle(vector1, vector2):
    #calc the angle between two vectors based on cosine theorem
    ##construct two array for vector calculation
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    ##calc the length of two vectors
    # s1 = np.sqrt(v1.dot(v1))
    s1 = np.sqrt(np.dot(v1, v1))
    # s2 = np.sqrt(v2.dot(v2))
    s2 = np.sqrt(np.dot(v2, v2))

    ##calc the angle of two vectors
    # cos_theta = v1.dot(v2) / (s1 * s2)
    cos_theta = np.dot(v1, v2) / (s1 * s2)
    vectors_angle = np.arccos(cos_theta)
    
    return vectors_angle


def calc_plane_normal_vector(key_point0, key_point1, key_point2):
    #calc the normal vector of a plane based on two vectors which constructed from three key points
    ##construct two vectors(kp0kp1, kp0kp2)
    vector1 = make_vector(key_point0, key_point1)
    vector2 = make_vector(key_point0, key_point2)

    ##calc the normal vector of the plane defined by the two vectors(vector1 cross product vector2)
    # plane_normal_vector = np.cross(vector1, vector2)
    vx = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    vy = vector1[2] * vector2[0] - vector1[0] * vector2[2]
    vz = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    length_vector = np.sqrt(vx * vx + vy * vy + vz * vz)
    nx = vx / length_vector
    ny = vy / length_vector
    nz = vz / length_vector

    ##construct the array for the normal vector
    plane_normal_vector = np.array([nx, ny, nz])

    return plane_normal_vector


def calc_vector_plane_angle(vector, plane_normal_vector):
    #calc the angle between the vector and the plane based on the vector and the normal vector of the plane according to cosine theorem
    ##calc the length of the two vector
    # length_vector = np.sqrt(vector.dot(vector))
    length_vector = np.sqrt(np.dot(vector, vector))
    # length_plane_normal_vector = np.sqrt(plane_normal_vector.dot(plane_normal_vector))
    length_plane_normal_vector = np.sqrt(np.dot(plane_normal_vector, plane_normal_vector))

    ##calc the angle between the vector and the plane
    # sin_theta = vector.dot(plane_normal_vector) / (length_vector * length_plane_normal_vector)
    sin_theta = np.dot(vector, plane_normal_vector) / (length_vector * length_plane_normal_vector)
    vector_plane_angle = np.arcsin(sin_theta)

    return vector_plane_angle


def calc_plane_plane_angle(plane_normal_vector1, plane_normal_vector2):
    #calc the angle between two plane based on the normal vectors of the two plane
    vectors_angle = calc_vectors_angle(plane_normal_vector1, plane_normal_vector2)

    #check the vectors_angle whether is over 90 degree and make sure that the plane_plane_angle is less than 90 degree
#     if vectors_angle <= np.pi / 2:
#         plane_plane_angle = np.pi / 2 - vectors_angle
#     elif vectors_angle > np.pi / 2:
#         plane_plane_angle = vectors_angle - np.pi / 2
    if vectors_angle <= np.pi / 2:
        plane_plane_angle = vectors_angle
    elif vectors_angle > np.pi / 2:
        plane_plane_angle = np.pi - vectors_angle
    
    return plane_plane_angle


def calc_vector_projection_onto_vector(vector, vector_ref):
    #calc the vector projection onto the reference vector_ref(Proj_n_u = u*n/|n|^2*n)
    # dot_product = vector.dot(vector_ref)
    dot_product = np.dot(vector, vector_ref)
    # vector_ref_length_square = vector_ref.dot(vector_ref)
    vector_ref_length_square = np.dot(vector_ref, vector_ref)
    
    vector_projection_onto_vector = dot_product / vector_ref_length_square * vector_ref

    return vector_projection_onto_vector


def calc_vector_projection_onto_plane(vector, plane_normal_vector):
    #calc the vector projection onto the plane based on the vector and the plane normal vector
    #(Proj_plane_u = u - Proj_n_u = u - u*n/|n|^2*n)
    #A(AtA)-1AtX
    ##calc the vector projection onto the normal vector of the plane
    vector_projection_onto_normal_vector = calc_vector_projection_onto_vector(vector, plane_normal_vector)

    ##calc the vector projection onto the plane
    vector_projection_onto_plane = vector - vector_projection_onto_normal_vector

    return vector_projection_onto_plane

