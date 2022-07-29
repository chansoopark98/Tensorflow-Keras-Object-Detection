
from model_configuration import ModelConfiguration
import tensorflow as tf
import argparse
import time


# 1. sudo apt-get install libtcmalloc-minimal4
# 2. check dir ! 
# dpkg -L libtcmalloc-minimal4
# 3. LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model",  help="SavedModel.pb 변환", action='store_true')
parser.add_argument("--saved_model_path", type=str,   help="Saved model weight path",
                    default='your_model_checkpoints')

# Set Training Options
parser.add_argument("--model_prefix",     type=str,    help="Model name",
                    default='test_mobilenet_ssd')
parser.add_argument("--batch_size",       type=int,    help="Batch size per each GPU",
                    default=32)
parser.add_argument("--epoch",            type=int,    help="Training epochs",
                    default=100)
parser.add_argument("--lr",               type=float,  help="Initial learning rate",
                    default=0.001)
parser.add_argument("--weight_decay",     type=float,  help="Set Weight Decay",
                    default=0.0005)
parser.add_argument("--image_size",       type=tuple,  help="Set model input size",
                    default=(300, 300))
parser.add_argument("--optimizer",        type=str,    help="Set optimizer",
                    default='adam')
parser.add_argument("--use_weightDecay",  type=bool,   help="Whether to use weightDecay",
                    default=False)
parser.add_argument("--mixed_precision",  type=bool,   help="Whether to use mixed_precision",
                    default=True)
parser.add_argument("--model_name",       type=str,    help="Set the model name to save",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))

# Set directory path (Dataset,  Dataset_type, Chekcpoints, Tensorboard)
parser.add_argument("--dataset_dir",      type=str,    help="Set the dataset download directory",
                    default='./datasets/')
parser.add_argument("--dataset_name",      type=str,    help="Set the dataset type (cityscapes, custom etc..)",
                    default='voc')
parser.add_argument("--checkpoint_dir",   type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,    help="Set tensorboard storage path",
                    default='tensorboard/')

# Set Distribute training (When use Single gpu)
parser.add_argument("--gpu_num",          type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)

# Set Distribute training (When use Multi gpu)
parser.add_argument("--multi_gpu",  help="Set up distributed learning mode", action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    if args.saved_model:
        model = ModelConfiguration(args=args)
        model.saved_model()

    else:
        if args.multi_gpu == False:
            tf.config.set_soft_device_placement(True)

            gpu_number = '/device:GPU:' + str(args.gpu_num)
            with tf.device(gpu_number):
                model = ModelConfiguration(args=args)
                model.train()

        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = ModelConfiguration(args=args, mirrored_strategy=mirrored_strategy)
                model.train()




# WEIGHT_DECAY = args.weight_decay
# BATCH_SIZE = args.batch_size
# EPOCHS = args.epoch
# base_lr = args.lr
# SAVE_MODEL_NAME = args.model_name
# DATASET_DIR = args.dataset_dir
# CHECKPOINT_DIR = args.checkpoint_dir
# TENSORBOARD_DIR = args.tensorboard_dir
# MODEL_NAME = args.backbone_model
# TRAIN_MODE = args.train_dataset
# IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
# USE_WEIGHT_DECAY = args.use_weightDecay
# LOAD_WEIGHT = args.load_weight
# MIXED_PRECISION = args.mixed_precision
# DISTRIBUTION_MODE = args.distribution_mode

# os.makedirs(DATASET_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Matching target transform
# specs = set_priorBox(MODEL_NAME)
# priors = create_priors_boxes(specs, IMAGE_SIZE[0])
# TARGET_TRANSFORM = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# # Create Dataset
# dataset_config = GenerateDatasets(data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, target_transform=TARGET_TRANSFORM, dataset_name='voc')
# train_data = dataset_config.get_trainData(train_data=dataset_config.train_data)
# valid_data = dataset_config.get_validData(valid_data=dataset_config.valid_data)
# # Set loss function
# loss = Total_loss(dataset_config.num_classes)

# print("Backbone Network Version : EfficientNet{0} .".format(MODEL_NAME))

# steps_per_epoch = dataset_config.number_train // BATCH_SIZE
# validation_steps = dataset_config.number_test // BATCH_SIZE
# print("Train batch samples{0}".format(steps_per_epoch))
# print("Validation batch samples{0}".format(validation_steps))

# metrics = CreateMetrics(dataset_config.num_classes)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

# checkpoint = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '.h5',
#                                  monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
# testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)
# polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
#                                                           decay_steps=EPOCHS,
#                                                           end_learning_rate=0.0001, power=1.0)
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

# optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

# # Whether to use Mixed Precision
# if MIXED_PRECISION:
#     policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
#     mixed_precision.set_policy(policy)
#     optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

# callback = [checkpoint, tensorboard, testCallBack, lr_scheduler]

# if DISTRIBUTION_MODE == 'multi':
#     mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#         tf.distribute.experimental.CollectiveCommunication.NCCL)

# else:
#     mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

    
# # Model builder
# model = model_build(TRAIN_MODE, MODEL_NAME, normalizations=normalize, num_priors=num_priors,
#                     image_size=IMAGE_SIZE, backbone_trainable=True)

# # Model summary
# model.summary()

# if USE_WEIGHT_DECAY:
#     regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY / 2)
#     for layer in model.layers:
#         for attr in ['kernel_regularizer', 'bias_regularizer']:
#             if hasattr(layer, attr) and layer.trainable:
#                 setattr(layer, attr, regularizer)

# # Model compile
# model.compile(
#     optimizer=optimizer,
#     loss=loss.total_loss,
#     metrics=[metrics.precision, metrics.recall, metrics.cross_entropy, metrics.localization]
# )

# # If you use pre-trained model or resume training
# if LOAD_WEIGHT:
#     weight_name = 'voc_0710'
#     model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

# # Start train
# history = model.fit(train_data,
#         validation_data=valid_data,
#         steps_per_epoch=steps_per_epoch,
#         validation_steps=validation_steps,
#         epochs=EPOCHS,
#         callbacks=callback)

# # Model save after training
# model.save('./checkpoints/save_model.h5', True, True, 'h5')

