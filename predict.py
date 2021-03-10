from utils.priors import *
from model.model_builder import ssd
from utils.model_post_processing import post_process
from utils.misc import color_map
import os
from preprocessing import prepare_for_prediction
from tqdm import tqdm
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=32)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0309.h5')
parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./inputs/')
parser.add_argument("--output_dir", type=str,   help="테스트 결과 이미지 디렉토리 설정", default='./outputs/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')



args = parser.parse_args()
BATCH_SIZE = args.batch_size
IMAGE_SIZE = [args.image_size, args.image_size]
DATASET_DIR = args.dataset_dir
checkpoint_filepath = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
                Spec(48, 8, BoxSizes(38, 77), [2]), # 0.1
                Spec(24, 16, BoxSizes(77, 142), [2, 3]), # 0.2
                Spec(12, 32, BoxSizes(142, 207), [2, 3]), # 0.37
                Spec(6, 64, BoxSizes(207, 273), [2, 3]), # 0.54
                Spec(3, 128, BoxSizes(273, 337), [2]), # 0.71
                Spec(1, 384, BoxSizes(337, 403), [2]) # 0.88 , max 1.05
        ]

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(TRAIN_MODE ,MODEL_NAME, pretrained=False)
model.summary()
print("모델로드")
model.load_weights(checkpoint_filepath)

filenames = os.listdir(INPUT_DIR)
filenames.sort()
dataset = tf.data.Dataset.list_files(INPUT_DIR + '/*', shuffle=False)
dataset = dataset.map(prepare_for_prediction(IMAGE_SIZE))
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
        _, color = color_map(int(labels[i]-1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_box, CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)


for batch in tqdm(dataset, total=test_steps):

    pred = model.predict_on_batch(batch)
    predictions = post_process(pred, target_transform, confidence_threshold=0.4)

    for i, path in enumerate(filenames[x:y]):

        im = cv2.imread(INPUT_DIR+'/'+path)
        pred_boxes, pred_scores, pred_labels = predictions[i]
        if pred_boxes.size > 0:

            draw_bounding(im, pred_boxes,  labels=pred_labels, img_size=im.shape[:2])
        fn = OUTPUT_DIR + '/' + path +'.jpg'
        cv2.imwrite(fn, im)

    x = y
    y += BATCH_SIZE
