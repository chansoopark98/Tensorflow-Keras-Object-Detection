import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
from model.loss import multibox_loss
import os
from preprocessing import prepare_dataset
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.utils import plot_model

#from keras.utils.vis_utils import plot_model
#from keras.utils import plot_model

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
"""실험일지
0928 efficientnetb0_SSD를 BATCH 16 EPOCH 100으로 추가 학습 한 결과 EVAL MAP는 약 74%
0929 오전1:18 DEFAULT BOX SCALE 수정 후(38,38 ASPECT RATIO 변경 2 -> 2,3 총 6) pretrain값은 4,6,6,6,4,4로 고정되어 있기 때문에 pretrain이 불가함
"""

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


DATASET_DIR = './datasets/'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
EPOCHS = 200
checkpoint_filepath = './checkpoints/'
base_lr = 1e-3 if checkpoint_filepath is None else 1e-5

train2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='train')
valid2012 = tfds.load('voc/2012', data_dir=DATASET_DIR, split='validation')

train2007 = tfds.load("voc", data_dir=DATASET_DIR, split='train')
valid2007 = tfds.load("voc", data_dir=DATASET_DIR, split='validation')

train_data = train2007.concatenate(valid2007).concatenate(train2012).concatenate(valid2012)

number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
# number_train = 5011
print("Number of Training Files:", number_train)

test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
# number_test = 4952
print("Number of Test Files:", number_test)


iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2
# train.py에서 priors를 변경하면 model/ssd.py도 수정해야함
specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2, 3]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
        ]


priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

# 데이터세트 인스턴스화 (input은 300x300@3 labels은 8732)
training_dataset = prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=True)
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(MODEL_NAME)
steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)
model.summary()

#flops = get_flops(model, BATCH_SIZE)
#print(f"FLOPS: {flops}")

#plot_model(model,'model_b0_ssd.png',show_shapes=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_filepath+'0112_home.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
    loss = multibox_loss
)



history = model.fit(training_dataset, 
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr,checkpoint])
save_path = './checkpoints/'
model.save(save_path, save_format='h5')