import tensorflow as tf

class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, y_true, y_pred):
        loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))
        return loss
