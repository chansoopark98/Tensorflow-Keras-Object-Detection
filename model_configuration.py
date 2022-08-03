from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
import os
from utils.priors import *
from utils.load_datasets import GenerateDatasets
from utils.metrics import CreateMetrics
from model.model_builder import ModelBuilder
from model.loss import DetectionLoss
# from model.test_loss import DetectionLoss


class ModelConfiguration(GenerateDatasets):
    def __init__(self, args: argparse, mirrored_strategy: object = None):
        """
        Args:
            args (argparse): Training options (argparse).
            mirrored_strategy (tf.distribute): tf.distribute.MirroredStrategy() with Session.
        """
        self.args = args
        self.mirrored_strategy = mirrored_strategy
        self.check_directory(dataset_dir=args.dataset_dir,
                             checkpoint_dir=args.checkpoint_dir, model_name=args.model_name)
        self.configuration_args()
        self.target_transform = self.configuration_transforms()

        super().__init__(data_dir=self.DATASET_DIR,
                         image_size=self.IMAGE_SIZE,
                         batch_size=self.BATCH_SIZE,
                         image_norm_type=args.image_norm_type,
                         target_transform=self.target_transform,
                         dataset_name=args.dataset_name)


    def check_directory(self, dataset_dir: str, checkpoint_dir: str, model_name: str) -> None:
        """
        Args:
            dataset_dir    (str) : Tensorflow dataset directory.
            checkpoint_dir (str) : Directory to store training weights.
            model_name     (str) : Model name to save.
        """
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir + model_name, exist_ok=True)
    

    def configuration_args(self):
        """
            Set training variables from argparse's arguments 
        """
        self.MODEL_PREFIX = self.args.model_prefix
        self.BACKBONE_NAME = self.args.backbone_name
        self.WEIGHT_DECAY = self.args.weight_decay
        self.OPTIMIZER_TYPE = self.args.optimizer
        self.BATCH_SIZE = self.args.batch_size
        self.EPOCHS = self.args.epoch
        self.INIT_LR = self.args.lr
        self.SAVE_MODEL_NAME = self.args.model_name + '_' + self.args.model_prefix
        self.DATASET_DIR = self.args.dataset_dir
        self.DATASET_NAME = self.args.dataset_name
        self.CHECKPOINT_DIR = self.args.checkpoint_dir
        self.TENSORBOARD_DIR = self.args.tensorboard_dir
        self.IMAGE_SIZE = self.args.image_size
        self.USE_WEIGHT_DECAY = self.args.use_weightDecay
        self.MIXED_PRECISION = self.args.mixed_precision
        self.DISTRIBUTION_MODE = self.args.multi_gpu
        if self.DISTRIBUTION_MODE:
            self.BATCH_SIZE *= 2

    def configuration_dataset(self) -> None:
        """
            Configure the dataset. Train and validation dataset is inherited from the parent class and used.
        """
        # Wrapping tf.data generator
        self.train_data = self.get_trainData(train_data=self.train_data)
        self.valid_data = self.get_validData(valid_data=self.valid_data)
    
        # Calculate training and validation steps
        self.steps_per_epoch = self.number_train // self.BATCH_SIZE
        self.validation_steps = self.number_valid // self.BATCH_SIZE

        # Wrapping tf.data generator if when use multi-gpu training
        if self.DISTRIBUTION_MODE:
            self.train_data = self.mirrored_strategy.experimental_distribute_dataset(self.train_data)
            self.valid_data = self.mirrored_strategy.experimental_distribute_dataset(self.valid_data)   


    def configuration_transforms(self) -> object:
        spec_list = convert_spec_list()
        priors = create_priors_boxes(specs=spec_list, image_size=self.IMAGE_SIZE[0], clamp=True)
        target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

        return target_transform


    def __set_callbacks(self):
        """
            Set the keras callback of model.fit.

            For some metric callbacks, the name of the custom metric may be different and may not be valid,
            so you must specify the name of the custom metric.
        """
        # Set training keras callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)
        
        checkpoint_val_loss = ModelCheckpoint(self.CHECKPOINT_DIR + self.args.model_name + '/_' + self.SAVE_MODEL_NAME + '_best_loss.h5',
                                              monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

        tensorboard = TensorBoard(log_dir=self.TENSORBOARD_DIR + 'detection/' +
                                  self.MODEL_PREFIX, write_graph=True, write_images=True)

        polyDecay = PolynomialDecay(initial_learning_rate=self.INIT_LR,
                                                                  decay_steps=self.EPOCHS,
                                                                  end_learning_rate=self.INIT_LR * 0.01, power=0.9)

        lr_scheduler = LearningRateScheduler(polyDecay, verbose=1)
        
        # If you wanna need another callbacks, please add here.
        self.callback = [checkpoint_val_loss,  tensorboard, lr_scheduler]

    
    def __set_optimizer(self):
        """
            Configure the optimizer for backpropagation calculations.
        """
        if self.OPTIMIZER_TYPE == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.INIT_LR)
        elif self.OPTIMIZER_TYPE == 'radam':
            self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.INIT_LR,
                                                          weight_decay=0.00001,
                                                          total_steps=int(
                                                          self.number_train / (self.BATCH_SIZE / self.EPOCHS)),
                                                          warmup_proportion=0.1,
                                                          min_lr=0.0001)
        if self.MIXED_PRECISION:
            # Wrapping optimizer when use distribute training (multi-gpu training)
            mixed_precision.set_global_policy('mixed_float16')
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    def __set_metrics(self):
        metric = CreateMetrics(num_classes=self.num_classes)
        metrics = None

        return metrics

        
    def __configuration_model(self):
        """
            Build a deep learning model.
        """
        self.model = ModelBuilder(image_size=self.IMAGE_SIZE,
                                  num_classes=self.num_classes).build_model(model_name=self.BACKBONE_NAME)
        # self.model.load_weights('./checkpoints/0802/_0802_efficientv2b3_new_display_dataset_remove_rotation_best_loss.h5', by_name=True, skip_mismatch=True)

    
    def train(self):
        """
            Compile all configuration settings required for training.
            If the custom metric name is different in the __set_callbacks function,
            the update may not be possible, so please check the name.
        """
        self.configuration_dataset()
        self.__set_optimizer()
        self.metrics = self.__set_metrics()
        self.__set_callbacks()
        self.__configuration_model()

        self.loss = DetectionLoss(num_classes=self.num_classes,
                                  global_batch_size=self.batch_size,
                                  use_multi_gpu=self.DISTRIBUTION_MODE)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.model.summary()

        self.model.fit(self.train_data,
                       validation_data=self.valid_data,
                       steps_per_epoch=self.steps_per_epoch,
                       validation_steps=self.validation_steps,
                       epochs=self.EPOCHS,
                       callbacks=self.callback)



    def saved_model(self):
        """
            Convert it to a graph model (.pb) using the learned weights.
        """
        self.model = ModelBuilder(image_size=self.IMAGE_SIZE,
                                  num_classes=self.num_classes).build_model()
        self.model.load_weights(self.args.saved_model_path)
        export_path = os.path.join(self.CHECKPOINT_DIR, 'export_path', '1')
        
        os.makedirs(export_path, exist_ok=True)
        self.export_path = export_path

        self.model.summary()

        tf.keras.models.save_model(
            self.model,
            self.export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None
        )
        print("save model clear")