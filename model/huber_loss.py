import sys

import tensorflow as tf
import numpy as np
from tensorflow import keras

def smooth_l1(labels, scores, sigma=1.0):
    diff = scores-labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))

def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_ratio

    loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss)

    indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
    orders = tf.argsort(indexes, axis=1)
    neg_mask = tf.cast(orders, tf.float32) < num_neg

    return tf.logical_or(pos_mask ,neg_mask)


def total_loss(y_true, y_pred, num_classes=21):
    labels = tf.argmax(y_true[:,:,:num_classes], axis=2)
    f_labels = y_true[:, :, :num_classes]
    # f_labels = tf.cast(tf.argmax(y_true[:,:,:num_classes], axis=2), tf.float32)
    confidence = y_pred[:,:,:num_classes]
    predicted_locations = y_pred[:,:,num_classes:]
    gt_locations = y_true[:,:,num_classes:]

    alpha = 0.25
    gamma = 1.5
    #indices = labels > 0
    indices = f_labels > 0
    f_labels = tf.boolean_mask(f_labels, indices)

    classification = tf.boolean_mask(confidence, indices)
    #f_labels = tf.gather_nd(labels, indices)
    #classification = tf.gather_nd(confidence, indices)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(f_labels) * alpha
    tf.print(alpha_factor, sys.stdout)
    alpha_factor = tf.where(keras.backend.equal(f_labels, 1), alpha_factor, 1 - alpha_factor)
    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    focal_weight = tf.where(keras.backend.equal(f_labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma


    classification_loss= focal_weight * keras.backend.binary_crossentropy(f_labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(keras.backend.equal(f_labels, 1))
    normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

    classification_loss = keras.backend.sum(classification_loss) / normalizer

    # neg_pos_ratio = 3.0
    # loss = -tf.nn.log_softmax(confidence, axis=2)[:, :, 0]
    # loss = tf.stop_gradient(loss)
    #
    # mask = hard_negative_mining(loss, labels, neg_pos_ratio)
    # mask = tf.stop_gradient(mask) # neg sample 마스크
    #
    # confidence = tf.boolean_mask(confidence, mask)
    # # calc classification loss
    # classification_loss = tf.math.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = tf.reshape(confidence, [-1, num_classes]),
    #                                                                                         labels = tf.boolean_mask(labels, mask)))

    pos_mask = labels > 0
    predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
    gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])
    # calc localization loss
    smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations,labels=gt_locations))


    num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
    # divide num_pos objects
    loc_loss = smooth_l1_loss / num_pos

    mbox_loss = loc_loss + classification_loss
    return mbox_loss