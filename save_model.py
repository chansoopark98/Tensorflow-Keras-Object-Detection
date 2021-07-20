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
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=300)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.01)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='CSNet-tiny')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=True)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
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
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
DISTRIBUTION_MODE = args.distribution_mode

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

specs = set_priorBox(MODEL_NAME)

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
TARGET_TRANSFORM = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# Create Dataset
dataset_config = GenerateDatasets(TRAIN_MODE, DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, TARGET_TRANSFORM)

# Set loss function
loss = Total_loss(dataset_config.num_classes)

print("백본 EfficientNet{0} .".format(MODEL_NAME))

steps_per_epoch = dataset_config.number_train // BATCH_SIZE
validation_steps = dataset_config.number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)

metrics = CreateMetrics(dataset_config.num_classes)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + TRAIN_MODE + '_' + SAVE_MODEL_NAME + '.h5',
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)


polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.0001, power=0.5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay)

# optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

callback = [checkpoint, tensorboard, testCallBack, lr_scheduler]

if MODEL_NAME == 'CSNet-tiny':
    normalize = [-1, -1, -1, -1, -1, -1]
    num_priors = [3, 3, 3, 3, 3, 3]
else :
    normalize = [20, 20, 20, -1, -1]
    num_priors = [3, 3, 3, 3, 3]

model = model_build(TRAIN_MODE, MODEL_NAME, normalizations=normalize, num_priors=num_priors,
                    image_size=IMAGE_SIZE, backbone_trainable=True)

model.compile(
    optimizer=optimizer,
    loss=loss.total_loss,
    metrics=[metrics.precision, metrics.recall, metrics.cross_entropy, metrics.localization]
)

weight_name = 'voc_0720'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')
model.summary()

# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#
#
#
# ret, frame = capture.read()
# #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
# x = tf.convert_to_tensor(frame)
#
# # 이미지 리사이징
# x = tf.image.resize(x, [224, 224])
# x = preprocess_input(x, mode='torch')
# x = tf.expand_dims(x, axis=0)
#
# start = time.perf_counter_ns()
# duration = (time.perf_counter_ns() - start)
# print(f"추론 과정 : {duration // 1000000}ms.")


""" convert to tflite """
model.save('./checkpoints/save_model', True, False, 'tf')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #converter.representative_dataset = representative_data_gen
converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the model.
with open('new_tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)

