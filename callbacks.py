import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, name, TENSORBOARD_DIR):
        super().__init__()
        self.name = name
        self.previous_loss = None
        self.file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
        self.file_writer.set_as_default()

    # def on_epoch_begin(self, epoch, logs=None):
    #     print("start epoch")
    #     print(self.model.optimizer.lr)


    def on_epoch_end(self, epoch, logs=None):
        logs['learning rate'] = self.model.optimizer.lr
        tf.summary.scalar("end_lr", logs['learning rate'], step=epoch)
    #
    #
    #     #self.previous_loss = logs['loss']
    #
    # def on_train_batch_begin(self, batch, logs=None):
    #     logs['learning rate'] = self.model.optimizer.lr
    #     # tf.summary.scalar("my_metric", logs['learning rate'], step=batch)
    # #
    # # def on_train_batch_end(self, batch, logs=None):
    # #

