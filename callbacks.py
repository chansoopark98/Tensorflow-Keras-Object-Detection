import tensorflow as tf
import gc

class Scalar_LR(tf.keras.callbacks.Callback):
    def __init__(self, name, TENSORBOARD_DIR):
        super().__init__()
        self.name = name
        # self.previous_loss = None
        self.file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
        self.file_writer.set_as_default()


    def on_epoch_end(self, epoch, logs=None):
        logs['learning rate'] = self.model.optimizer.lr
        tf.summary.scalar("end_lr", logs['learning rate'], step=epoch)
        tf.keras.backend.clear_session()
        gc.collect()


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


