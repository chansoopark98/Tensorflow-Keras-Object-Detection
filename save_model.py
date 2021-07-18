from tensorflow.keras.mixed_precision import experimental as mixed_precision
from config import *
from model.model_builder import model_build
import argparse
import time
import os

tf.keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='CSNet-tiny')

args = parser.parse_args()

TRAIN_MODE = 'voc'
CHECKPOINT_DIR = args.checkpoint_dir
TIME = args.model_name
MODEL_NAME = args.backbone_model
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]

TFLITE_DIR = CHECKPOINT_DIR + '/tflite/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TFLITE_DIR, exist_ok=True)

specs = set_priorBox(MODEL_NAME)

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
TARGET_TRANSFORM = MatchingPriors(priors, center_variance, size_variance, iou_threshold)


model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE, backbone_trainable=True)

weight_name = 'voc_0717'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.save(TFLITE_DIR+'/keras_model.h5' , True, True,'h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
converter.experimental_new_converter =True

tflite_model = converter.convert()
# Save the model.
with open(TFLITE_DIR+'tf_lite_model.tflite', 'wb') as f:
    f.write(tflite_model)


