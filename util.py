import numpy as np

def flip_aug(input_data):
    """
    input_data: the data to be augmented
    return: augmented data set
    """
    flipped = np.flip(input_data, -2)
    ret = np.vstack([input_data, flipped])
    return ret

def rot_aug(input_data):
    """
    input_data: the data to be augmented
    return: augmented data set
    """
    rot1 = np.rot90(input_data, axes=[-3,-2])
    rot2 = np.rot90(rot1, axes=[-3,-2])
    rot3 = np.rot90(rot2, axes=[-3,-2])
    ret = np.vstack([input_data, rot1, rot2, rot3])
    return ret

def expand_to_times(input_data, times):
    ret = input_data
    for i in range(times - 1):
        ret = np.vstack([ret, input_data])
    return ret
