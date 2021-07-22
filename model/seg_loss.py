import tensorflow as tf


def seg_loss(y_true, y_pred):
    # loss = tf.math.reduce_sum(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         logits = y_pred,
    #         labels = y_true)
    #     )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    loss = tf.reduce_sum(loss * (1. / 1))
    return loss
