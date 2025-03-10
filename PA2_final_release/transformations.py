import math
import numpy as np

def get_rot_mx(angle):
    '''
    Input:
        angle -- Rotation angle in radians
    Output:
        A 3x3 numpy array representing 2D rotations.
    '''
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 2
        elements
    Output:
        A 3x3 numpy array representing 2D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    return np.array([
        [1, 0, trans_vec[0]],
        [0, 1, trans_vec[1]],
        [0, 0, 1]
    ])

def get_scale_mx(s_x, s_y):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
    Output:
        A 3x3 numpy array representing 2D scaling.
    '''
    return np.array([
        [s_x, 0, 0],
        [0, s_y, 0],
        [0, 0, 1]
    ])

