import numpy as np

def expand_to_times(input_data, times):
    ret = input_data
    for i in range(times - 1):
        ret = np.vstack([ret, input_data])
    return ret

def flip_aug(input_data, label):
    """
    input_data: the data to be augmented
    return: augmented data set
    """
    flipped = np.flip(input_data, -2)
    ret_x = np.vstack([input_data, flipped])
    ret_label = expand_to_times(label, 2)
    return [ret_x, ret_label]

def rot_aug(input_data, label):
    """
    input_data: the data to be augmented
    return: augmented data set
    """
    rot1 = np.rot90(input_data, axes=[-3,-2])
    rot2 = np.rot90(rot1, axes=[-3,-2])
    rot3 = np.rot90(rot2, axes=[-3,-2])
    ret_x = np.vstack([input_data, rot1, rot2, rot3])
    ret_label = expand_to_times(label, 4)
    return [ret_x, ret_label]


