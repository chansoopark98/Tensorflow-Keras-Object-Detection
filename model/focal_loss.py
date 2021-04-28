import tensorflow as tf
import numpy as np
from tensorflow import keras

def smooth_l1(labels, scores, sigma=1.0):
    diff = scores-labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))

def hard_negative_mining(loss, labels, neg_pos_factor):
    pos_mask = labels > 0
    num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_factor
    loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss)
    indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
    orders = tf.argsort(indexes, axis=1)
    neg_mask = tf.cast(orders, tf.float32) < num_neg
    return tf.logical_or(pos_mask ,neg_mask)

def total_loss(y_true, y_pred, num_classes=81):
    labels = tf.argmax(y_true[:,:,:num_classes], axis=2) # batch, 13792
    focal_labels = y_true[:,:,:num_classes]
    confidence = y_pred[:,:,:num_classes] # batch, None, 81
    predicted_locations = y_pred[:,:,num_classes:] # None, None, 4
    gt_locations = y_true[:,:,num_classes:] # None, 13792, None

    # compute the focal loss
    alpha = 0.25
    gamma = 1.5

    labels = tf.argmax(y_true[:, :, :-1], axis=2)
    # labels = y_true[:, :, :-1]
    # -1 for ignore, 0 for background, 1 for object
    anchor_state = y_true[:, :, num_classes:]
    classification = y_pred

    indices = tf.where(keras.backend.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)
    # cls_loss = keras.backend.binary_crossentropy(labels, classification)

    # # compute the normalizer: the number of positive anchors
    # normalizer = tf.where(keras.backend.equal(anchor_state, 1))
    # normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    # normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)
    #
    # cls_loss = keras.backend.sum(cls_loss) / normalizer
    mbox_loss = cls_loss


    # pos_mask = labels > 0
    #
    # predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
    #
    # gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])
    #
    # smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations,labels=gt_locations))
    #
    # num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
    #
    # loc_loss = smooth_l1_loss / num_pos
    # class_loss = cls_loss
    # mbox_loss = loc_loss + class_loss

    return mbox_loss
