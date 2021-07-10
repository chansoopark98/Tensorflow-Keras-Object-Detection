from model.model_builder import model_build
from utils.model_post_processing import post_process
import os
from preprocessing import prepare_for_prediction
from tqdm import tqdm
import cv2
import argparse
from config import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications.imagenet_utils import preprocess_input
tf.keras.backend.clear_session()

policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=2)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0615_b0_mAP81.9%_voc.h5')
# parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./datasets/test/VOCdevkit/VOC2007/JPEGImages/')
parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./inputs/')
parser.add_argument("--output_dir", type=str,   help="테스트 결과 이미지 디렉토리 설정", default='./outputs/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')



args = parser.parse_args()
BATCH_SIZE = args.batch_size
MODEL_NAME = args.backbone_model
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
DATASET_DIR = args.dataset_dir
checkpoint_filepath = args.checkpoint_dir
TRAIN_MODE = args.train_dataset
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

if TRAIN_MODE == 'voc':
    CLASSES_NUM = 21
    CLASSES_LABEL = CLASSES
else:
    CLASSES_NUM = 81
    CLASSES_LABEL = COCO_CLASSES

specs = set_priorBox(MODEL_NAME)

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE, pretrained=False)
model.summary()
print("모델로드")
model.load_weights(checkpoint_filepath)



def draw_bounding(img , bboxes, labels, img_size):
    # resizing 작업
    if np.max(bboxes) < 10:

        bboxes[:, [0,2]] = bboxes[:, [0,2]]*img_size[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*img_size[0]

    for i, bbox in enumerate(bboxes):
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        img_box = np.copy(img)
        _, color = coco_color_map(int(labels[i] - 1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_box, CLASSES_LABEL[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()

    # 텐서 변환
    input = tf.convert_to_tensor(frame)
    #input = tf.image.decode_jpeg(input, channels=3)
    # 이미지 리사이징
    input = tf.image.resize(input, [512, 512])
    input = preprocess_input(input, mode='torch')
    input = tf.expand_dims(input, axis=0)

    pred = model.predict_on_batch(input)
    predictions = post_process(pred, target_transform, classes=CLASSES_NUM, confidence_threshold=0.4)

    pred_boxes, pred_scores, pred_labels = predictions[0]
    if pred_boxes.size > 0:
        draw_bounding(frame, pred_boxes, labels=pred_labels, img_size=frame.shape[:2])
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()