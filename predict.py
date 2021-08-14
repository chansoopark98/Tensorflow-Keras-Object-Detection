from model.model_builder import model_build
from utils.model_post_processing import post_process
import os
from preprocessing import prepare_for_prediction
from tqdm import tqdm
import cv2
import argparse
from config import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
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


os.makedirs(OUTPUT_DIR, exist_ok=True)


specs = set_priorBox(MODEL_NAME)


priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

if MODEL_NAME == 'CSNet-tiny':
    normalize = [-1, -1, -1, -1, -1, -1]
    num_priors = [3, 3, 3, 3, 3, 3]
else :
    normalize = [20, 20, 20, -1, -1]
    num_priors = [3, 3, 3, 3, 3]

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(TRAIN_MODE, MODEL_NAME, normalizations=normalize, num_priors=num_priors,
                    image_size=IMAGE_SIZE, backbone_trainable=False)
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

total_time=0

for batch in tqdm(dataset, total=test_steps):

    start = time.perf_counter_ns()
    pred = model.predict_on_batch(batch)

    #feature = model.get_layer('block5c_add').output
    feature = model.layers[13].output
    plt.imshow(feature[0].numpy())
    plt.show()

    duration = (time.perf_counter_ns() - start) / BATCH_SIZE
    print(f"inference time : {duration // 1000000}ms.")

    predictions = post_process(pred, target_transform, classes=CLASSES_NUM, confidence_threshold=0.2)


    for i, path in enumerate(filenames[x:y]):

        im = cv2.imread(INPUT_DIR+'/'+path)
        pred_boxes, pred_scores, pred_labels = predictions[i]
        if pred_boxes.size > 0:

            draw_bounding(im, pred_boxes,  labels=pred_labels, img_size=im.shape[:2])
        fn = OUTPUT_DIR + '/' + path +'.jpg'
        cv2.imwrite(fn, im)

    x = y
    y += BATCH_SIZE

print("속도 측정 :", total_time/4952 , " 단위(ms)")
