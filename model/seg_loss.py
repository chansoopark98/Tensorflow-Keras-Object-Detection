import tensorflow as tf
import numpy as np
class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, labels, logits):
        #logits = tf.reshape(logits, [-1, 20])
        labels = tf.cast(labels, dtype=tf.int64)
        labels = tf.squeeze(labels, -1)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

        loss = tf.math.reduce_mean(ce)

        # loss = tf.math.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, logits))

        return loss


