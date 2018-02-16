import numpy as np

def flip_aug(input_data):
    """
    input_data: the data to be augmented
    return: augmented data set
    """
    flipped = np.flip(input_data, -2)
    ret = np.vstack([input_data, flipped])
    return ret
