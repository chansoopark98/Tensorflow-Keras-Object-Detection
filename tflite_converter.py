from config import *
from model.model_builder import model_build
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/voc_0719.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='CSNet-tiny')

args = parser.parse_args()


CHECKPOINT_DIR = args.checkpoint_dir

MODEL_NAME = args.backbone_model
TRAIN_MODE = 'voc'
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]

if MODEL_NAME == 'CSNet-tiny':
    normalize = [-1, -1, -1, -1, -1, -1]
    num_priors = [3, 3, 3, 3, 3, 3]
else :
    normalize = [20, 20, 20, -1, -1]
    num_priors = [3, 3, 3, 3, 3]

model = model_build(TRAIN_MODE, MODEL_NAME, normalizations=normalize, num_priors=num_priors,
                    image_size=IMAGE_SIZE, backbone_trainable=False)

model.load_weights(CHECKPOINT_DIR)
model.save('./checkpoints/save_model.h5', True, True, 'h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('TFLITE'+'_'+str(time.strftime('%m%d', time.localtime(time.time())))+'.tflite', 'wb') as f:
    f.write(tflite_model)