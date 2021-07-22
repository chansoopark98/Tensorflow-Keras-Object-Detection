import tensorflow as tf
import sys
class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, y_true, y_pred):
        y_pred = tf.nn.log_softmax(y_pred, axis=-1)
        loss = tf.math.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred))
        tf.print(loss, sys.stdout)
        #loss = loss * (1. / self.batch_size)
        return loss
