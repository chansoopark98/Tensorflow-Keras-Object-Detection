import tensorflow as tf
import psutil

class Scalar_LR(tf.keras.callbacks.Callback):
    def __init__(self, name, TENSORBOARD_DIR):
        super().__init__()
        self.name = name
        # self.previous_loss = None
        self.file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
        self.file_writer.set_as_default()


    def on_epoch_end(self, epoch, logs=None):
        logs['learning rate'] = self.model.optimizer.lr
        # logs['weight decay'] = self.model.optimizer.weight_decay
        tf.summary.scalar("learning rate", logs['learning rate'], step=epoch)
        # tf.summary.scalar("weight_decay", logs['weight decay'], step=epoch)
        print(psutil.virtual_memory().used / 2 ** 30)


class DecayHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.lr = []
        self.wd = []
    def on_batch_end(self, batch, logs={}):
        self.lr.append(self.model.optimizer.lr(self.model.optimizer.iterations))
        self.wd.append(self.model.optimizer.weight_decay)


    def on_epoch_end(self, epoch, logs={}):
        print("end_batch lr : ",self.model.optimizer.lr)
        print("end_batch wd : ",self.model.optimizer.weight_decay)


