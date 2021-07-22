import tensorflow as tf

class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, y_true, y_pred):
        y_pred = tf.nn.log_softmax(y_pred, axis=-1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred)

        loss = loss * (1. / self.batch_size)
        return loss
