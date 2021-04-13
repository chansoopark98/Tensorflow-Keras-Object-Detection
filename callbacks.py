import tensorflow as tf

class Scalar_LR(tf.keras.callbacks.Callback):
    def __init__(self, name, TENSORBOARD_DIR):
        super().__init__()
        self.name = name
        # self.previous_loss = None
        self.file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
        self.file_writer.set_as_default()

    # def on_epoch_begin(self, epoch, logs=None):
    #     logs['learning rate'] = self.model.optimizer.lr
    #     tf.summary.scalar("lr", logs['learning rate'], step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs['learning rate'] = self.model.optimizer.lr
        # with self.file_writer.as_default():
        #     # img = self.model.predict(dummy_data)
        #     # y_pred = self.model.predict(self.validation_data[0])
        #     tf.summary.image("Training data", img, step=0)
        tf.summary.scalar("end_lr", logs['learning rate'], step=epoch)

    #
    #
    #     #self.previous_loss = logs['loss']
    #
    # def on_train_batch_begin(self, batch, logs=None):
    #     logs['learning rate'] = self.model.optimizer.lr
    #     # tf.summary.scalar("my_metric", logs['learning rate'], step=batch)
    # #
    # def on_train_batch_end(self, batch, logs=None):
    #     print('test')
    #
    #     # tensor = self.model.get_layer('block3b_add').output
    #     # tensor = self.model.layers[0].output
    #     # tensor = tensor[0,:,:,:]
    #     # print(tensor)
    #     # plt.imshow(tensor)
    #     # plt.show()
    #
    #     # intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
    #     #                                  outputs=self.model.get_layer('block3b_add').output)
    #     # intermediate_output = intermediate_layer_model.predict(self.validation_data[0])
    #     # print(intermediate_output)
    #
    #     # output_images = tf.cast(self.model.call(self.data['image']),dtype=tf.float32)
    #     # output_images *= 255
    #     # print(output_images)
    #
    #     # tf.summary.image('test', tensor, step=batch, max_outputs=1)
    #
    #

