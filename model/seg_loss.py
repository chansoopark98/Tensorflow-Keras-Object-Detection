import tensorflow as tf

class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, y_true, y_pred):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        #loss = tf.reduce_sum(loss * (1. / self.batch_size))
        return loss


def seg_loss(y_true, y_pred):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_true, y_pred)
    #loss = tf.reduce_sum(loss * (1. / self.batch_size))
    return loss