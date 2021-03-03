import tensorflow_datasets as tfds
from utils.priors import *
import argparse
import time
from preprocessing import prepare_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from model.model_builder import ssd
from tensorflow.keras.utils import plot_model
from calc_flops import get_flops

import cProfile

#
# tf.debugging.experimental.enable_dump_debug_info(
#     "/tmp/tfdbg2_logdir",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=16)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')
parser.add_argument("--pretrain_mode",  type=bool,  help="저장되어 있는 가중치 로드", default=False)

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
IMAGE_SIZE = [args.image_size, args.image_size]
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
checkpoint_filepath = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CONTINUE_TRAINING = args.pretrain_mode



if TRAIN_MODE == 'voc':
    from model.pascal_loss import total_loss
    train_pascal_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
    valid_train_12 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

    train_pascal_07 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
    valid_train_07 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

    train_data = train_pascal_07.concatenate(valid_train_07).concatenate(train_pascal_12).concatenate(valid_train_12)

    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("학습 데이터 개수", number_train)
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

else :
    from model.coco_loss import total_loss
    train_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='train')
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    train_data = train_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    test_data = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation')
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    # number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    number_train = 117266
    print("학습 데이터 개수", number_train)
    # number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    number_test = 4952
    print("테스트 데이터 개수:", number_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)


iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

# specs = [
#                 Spec(48, 8, BoxSizes(40, 90), [2]),
#                 Spec(24, 16, BoxSizes(90, 151), [2, 3]),
#                 Spec(12, 32, BoxSizes(151, 212), [2, 3]),
#                 Spec(6, 64, BoxSizes(212, 273), [2, 3]),
#                 Spec(3, 128, BoxSizes(273, 334), [2]),
#                 Spec(1, 384, BoxSizes(334, 395), [2])
#         ]

specs = [
                Spec(48, 8, BoxSizes(38, 77), [2]), # 0.1
                Spec(24, 16, BoxSizes(77, 142), [2, 3]), # 0.2
                Spec(12, 32, BoxSizes(142, 207), [2, 3]), # 0.37
                Spec(6, 64, BoxSizes(207, 273), [2, 3]), # 0.54
                Spec(3, 128, BoxSizes(273, 337), [2]), # 0.71
                Spec(1, 384, BoxSizes(337, 403), [2]) # 0.88 , max 1.05
        ]



priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

# 데이터세트 인스턴스화 (input은 300x300@3 labels은 8732)
training_dataset = prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE, target_transform, TRAIN_MODE, train=True)
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, TRAIN_MODE, train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE)

if CONTINUE_TRAINING is True:
    model.load_weights(checkpoint_filepath+'0217_main'+'.h5')

steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)
model.summary()

#flops = get_flops(model, BATCH_SIZE)
#print(f"FLOPS: {flops}")

#plot_model(model,'model_b0.png',show_shapes=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_filepath+SAVE_MODEL_NAME+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
isNanCheck = tf.keras.callbacks.TerminateOnNaN()



model.compile(
    optimizer=optimizer,
    loss=total_loss
)

history = model.fit(training_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr,checkpoint,isNanCheck])

