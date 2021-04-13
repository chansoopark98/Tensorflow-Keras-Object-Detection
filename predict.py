from utils.priors import *
from model.model_builder import model_build
from utils.model_post_processing import post_process
from utils.misc import voc_color_map
import os
from preprocessing import prepare_for_prediction
from tqdm import tqdm
import cv2
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=32)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0410_81.9%_b1_/0410.h5')
# parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./datasets/test/VOCdevkit/VOC2007/JPEGImages/')
parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./inputs/')
parser.add_argument("--output_dir", type=str,   help="테스트 결과 이미지 디렉토리 설정", default='./outputs/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B1')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')

MODEL_INPUT_SIZE = {
    'B0': 512,
    'B1': 512,
    'B2': 640,
    'B3': 704,
    'B4': 768,
    'B5': 832,
    'B6': 896,
    'B7': 960
}
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
else:
    CLASSES_NUM = 81



os.makedirs(OUTPUT_DIR, exist_ok=True)

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
            Spec(int(IMAGE_SIZE[0]/16), int(IMAGE_SIZE[0]/32),
                 BoxSizes(int(IMAGE_SIZE[0]*0.1), int(IMAGE_SIZE[0]*0.24)), [2, 3]),  # 0.2
            Spec(int(IMAGE_SIZE[0]/32), int(IMAGE_SIZE[0]/16),
                 BoxSizes(int(IMAGE_SIZE[0]*0.24), int(IMAGE_SIZE[0]*0.37)), [2, 3]),  # 0.37
            Spec(int(IMAGE_SIZE[0]/64), int(IMAGE_SIZE[0]/8),
                 BoxSizes(int(IMAGE_SIZE[0]*0.45), int(IMAGE_SIZE[0]*0.58)), [2, 3]),  # 0.54
            Spec(int(IMAGE_SIZE[0]/128), int(IMAGE_SIZE[0]/4),
                 BoxSizes(int(IMAGE_SIZE[0]*0.6), int(IMAGE_SIZE[0]*0.76)), [2]),  # 0.71
            Spec(int(IMAGE_SIZE[0] / 256), int(IMAGE_SIZE[0]/2),
                 BoxSizes(int(IMAGE_SIZE[0] * 0.76), int(IMAGE_SIZE[0] * 0.9)), [2]) # 0.88 / 0.95
        ]


priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(TRAIN_MODE, MODEL_NAME, pretrained=False)
model.summary()
print("모델로드")
model.load_weights(checkpoint_filepath)

filenames = os.listdir(INPUT_DIR)
filenames.sort()
dataset = tf.data.Dataset.list_files(INPUT_DIR + '*', shuffle=False)
dataset = dataset.map(prepare_for_prediction)
dataset = dataset.batch(BATCH_SIZE)

x, y = 0, BATCH_SIZE
test_steps = 4952 // BATCH_SIZE + 1



def draw_bounding(img , bboxes, labels, img_size):
    # resizing 작업
    if np.max(bboxes) < 10:

        bboxes[:, [0,2]] = bboxes[:, [0,2]]*img_size[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*img_size[0]

    for i, bbox in enumerate(bboxes):
        xmin = tf.cast(bbox[0], dtype=tf.int32)
        ymin = tf.cast(bbox[1], dtype=tf.int32)
        xmax = tf.cast(bbox[2], dtype=tf.int32)
        ymax = tf.cast(bbox[3], dtype=tf.int32)
        img_box = np.copy(img)
        _, color = coco_color_map(int(labels[i] - 1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_box, CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)


for batch in tqdm(dataset, total=test_steps):

    pred = model.predict_on_batch(batch)
    predictions = post_process(pred, target_transform, classes=CLASSES_NUM, confidence_threshold=0.15)

    for i, path in enumerate(filenames[x:y]):

        im = cv2.imread(INPUT_DIR+'/'+path)
        pred_boxes, pred_scores, pred_labels = predictions[i]
        if pred_boxes.size > 0:

            draw_bounding(im, pred_boxes,  labels=pred_labels, img_size=im.shape[:2])
        fn = OUTPUT_DIR + '/' + path +'.jpg'
        cv2.imwrite(fn, im)

    x = y
    y += BATCH_SIZE
