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

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
