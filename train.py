from tkinter.tix import IMAGE
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from callbacks import Scalar_LR
from metrics import CreateMetrics
from config import *
from utils.load_datasets import GenerateDatasets
from model.model_builder import model_build
from model.loss import Total_loss
import argparse
import time
import os

tf.keras.backend.clear_session()
# tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="Set Batch Size", default=32)
parser.add_argument("--epoch",          type=int,   help="Set Train Epochs", default=100)
parser.add_argument("--lr",             type=float, help="Set Learning Rate", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Set Weight Decay", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="Set the model name to be saved",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="The directory path where the dataset is stored", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="The path to the directory where the model is stored", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="Path where TensorBoard will be stored", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet (backbone) model", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="Set the dataset to be used for training coco or voc", default='voc')
parser.add_argument("--use_weightDecay",  type=bool,  help="Whether to use weight decay", default=True)
parser.add_argument("--load_weight",  type=bool,  help="Use pre-train weight", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="Whether to use Mixed Precision", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="Set up distributed learning mode (mirror or multi)", default='mirror')

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Matching target transform
specs = set_priorBox(MODEL_NAME)
priors = create_priors_boxes(specs, IMAGE_SIZE[0])
TARGET_TRANSFORM = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# Create Dataset
dataset_config = GenerateDatasets(data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, target_transform=TARGET_TRANSFORM, dataset_name='voc')
train_data = dataset_config.get_trainData(train_data=dataset_config.train_data)
valid_data = dataset_config.get_validData(valid_data=dataset_config.valid_data)
# Set loss function
loss = Total_loss(dataset_config.num_classes)

print("Backbone Network Version : EfficientNet{0} .".format(MODEL_NAME))

steps_per_epoch = dataset_config.number_train // BATCH_SIZE
validation_steps = dataset_config.number_test // BATCH_SIZE
print("Train batch samples{0}".format(steps_per_epoch))
print("Validation batch samples{0}".format(validation_steps))

metrics = CreateMetrics(dataset_config.num_classes)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '.h5',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)
polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.0001, power=1.0)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

# Whether to use Mixed Precision
if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

callback = [checkpoint, tensorboard, testCallBack, lr_scheduler]

if DISTRIBUTION_MODE == 'multi':
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

else:
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

    
# Model builder
model = model_build(TRAIN_MODE, MODEL_NAME, normalizations=normalize, num_priors=num_priors,
                    image_size=IMAGE_SIZE, backbone_trainable=True)

# Model summary
model.summary()

if USE_WEIGHT_DECAY:
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY / 2)
    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, regularizer)

# Model compile
model.compile(
    optimizer=optimizer,
    loss=loss.total_loss,
    metrics=[metrics.precision, metrics.recall, metrics.cross_entropy, metrics.localization]
)

# If you use pre-trained model or resume training
if LOAD_WEIGHT:
    weight_name = 'voc_0710'
    model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

# Start train
history = model.fit(train_data,
        validation_data=valid_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callback)

# Model save after training
model.save('./checkpoints/save_model.h5', True, True, 'h5')

