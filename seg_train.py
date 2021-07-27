from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from callbacks import Scalar_LR
from utils.load_datasets import CityScapes
from metrics import MeanIOU
from model.model_builder import seg_model_build
from model.seg_loss import Seg_loss
import argparse
import time
import os
import tensorflow as tf
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay

#LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so python seg_train.py
#LD_PRELOAD=/path/to/libtcmalloc_minimal.so.4 python seg_train.py
tf.keras.backend.clear_session()
#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python seg_train.py
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=150)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='city')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=True)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정 mirror or multi", default='mirror')

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
IMAGE_SIZE = [512, 1024]
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Create Dataset
dataset_config = CityScapes(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE)

# Set loss function


print("백본 EfficientNet{0} .".format(MODEL_NAME))

steps_per_epoch = dataset_config.number_train // BATCH_SIZE
validation_steps = dataset_config.number_valid // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

checkpoint_val_loss = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '_best_loss.h5',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
checkpoint_val_miou = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '_best_miou.h5',
                                 monitor='val_mean_iou', save_best_only=True, save_weights_only=True, verbose=1, mode='max')
testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)
polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.0001, power=0.9)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

# AdamW = extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
# optimizer = AdamW(weight_decay=WEIGHT_DECAY, learning_rate=base_lr)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)


if MIXED_PRECISION:
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

callback = [checkpoint_val_loss, checkpoint_val_miou, tensorboard, testCallBack, lr_scheduler]

if DISTRIBUTION_MODE == 'multi':
    mirrored_strategy = tf.distribute.experimental.Mtf.math.reduce_meanultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

else:
    # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    mirrored_strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():

    miou = MeanIOU(20)
    loss = Seg_loss(BATCH_SIZE)
    model = seg_model_build(MODEL_NAME, pretrained=True, image_size=IMAGE_SIZE)


    if USE_WEIGHT_DECAY:
        regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY / 2)
        for layer in model.layers:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(layer, attr) and layer.trainable:
                    setattr(layer, attr, regularizer)

    model.compile(
        optimizer=optimizer,
        loss=loss.total_loss,
        metrics=[miou])

    if LOAD_WEIGHT:
        weight_name = 'city_0726_best_loss'
        model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

    model.summary()

    history = model.fit(dataset_config.training_dataset,
            validation_data=dataset_config.validation_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callback)

    model.save('./checkpoints/save_model.h5', True, True, 'h5')

