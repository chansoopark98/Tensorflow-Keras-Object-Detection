import tensorflow as tf

class Seg_loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def total_loss(self, labels, logits):
        #logits = tf.reshape(logits, [-1, 20])
        labels = tf.squeeze(labels, -1)
        #labels = tf.reshape(labels, [-1])

        loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))




        return loss


