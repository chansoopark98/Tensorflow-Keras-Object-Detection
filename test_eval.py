import tensorflow_datasets as tfds
from utils.priors import *
from model.model_builder import model_build
from preprocessing import prepare_dataset
from preprocessing import coco_eval_dataset
from utils.model_post_processing import post_process  #
from utils.model_evaluation import eval_detection_voc
from tensorflow.keras.utils import plot_model
from calc_flops import get_flops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from config import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm
from pprint import pprint
import csv
import json
import argparse
import os

tf.keras.backend.clear_session()

policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
mixed_precision.set_policy(policy)


parser = argparse.ArgumentParser()


parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=2)
# parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0410_81.9%_b1_/0410.h5')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/coco_0608.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B1')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')
parser.add_argument("--eval_testdev",  type=str,   help="COCO minival(val dataset) 평가", default=True)
parser.add_argument("--calc_flops",  type=str,   help="모델 FLOPS 계산", default=False)
args = parser.parse_args()


BATCH_SIZE = args.batch_size
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CALC_FLOPS = args.calc_flops
IMAGE_SIZE = [MODEL_INPUT_SIZE[MODEL_NAME], MODEL_INPUT_SIZE[MODEL_NAME]]
EVAL_TESTDEV = args.eval_testdev
os.makedirs(DATASET_DIR, exist_ok=True)

specs = set_priorBox(MODEL_NAME)
priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)


if TRAIN_MODE == 'voc':
    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    CLASSES_NUM = 21
    with open('./pascal_labels.txt') as f:
        CLASSES = f.read().splitlines()

    validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                                target_transform, TRAIN_MODE, train=False)

else:
    if EVAL_TESTDEV:
        test_data, test_info = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation', with_info=True)
        test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
        test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    else:
        test_dev_idList = []
        with open('datasets/image_info_test-dev2017.json', 'r') as f:
            json_data = json.load(f)
            a = json_data['images']
            for i in range(len(a)):
                test_dev_idList.append(int(a[i]['id']))

        test_dev_idList = tf.convert_to_tensor(test_dev_idList, dtype=tf.int64)

        test_data, test_info = tfds.load('coco/2017', data_dir=DATASET_DIR, split='test', with_info=True)
        test_data = test_data.filter(lambda x: tf.reduce_all(
            tf.not_equal(tf.math.count_nonzero(tf.equal(x['image/id'], test_dev_idList)),0)))

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    with open('./coco_labels.txt') as f:
        CLASSES = f.read().splitlines()
    CLASSES_NUM = 81
    validation_dataset = coco_eval_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                     target_transform, TRAIN_MODE, train=False)

    validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                                target_transform, CLASSES_NUM, train=False)

mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))


with mirrored_strategy.scope():

    print("백본 EfficientNet{0} .".format(MODEL_NAME))
    model = model_build(TRAIN_MODE, MODEL_NAME, image_size=IMAGE_SIZE, pretrained=False)

    print("모델 가중치 로드...")
    model.load_weights(CHECKPOINT_DIR)
    model.summary()

    test_steps = number_test // BATCH_SIZE + 1
    print("테스트 배치 개수 : ", test_steps)



    test_difficults = []
    use_07_metric = True
    test_bboxes = []
    test_labels = []

    for sample in test_data:
        label = sample['objects']['label'].numpy()
        bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

        is_difficult = sample['objects']['is_crowd'].numpy()
        test_difficults.append(is_difficult)
        test_bboxes.append(bbox)
        test_labels.append(label)

    print("Evaluating..")
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    for x, y in tqdm(validation_dataset, total=test_steps):
        pred = model.predict_on_batch(x)
        predictions = post_process(pred, target_transform, classes=CLASSES_NUM)
        for prediction in predictions:
            boxes, scores, labels = prediction
            pred_bboxes.append(boxes)
            pred_labels.append(labels.astype(int) - 1)
            pred_scores.append(scores)

    answer = eval_detection_voc(pred_bboxes=pred_bboxes,
                                pred_labels=pred_labels,
                                pred_scores=pred_scores,
                                gt_bboxes=test_bboxes,
                                gt_labels=test_labels,
                                gt_difficults=test_difficults,
                                use_07_metric=use_07_metric)
    # print("*"*100)
    print("AP 결과")
    ap_dict = dict(zip(CLASSES, answer['ap']))
    pprint(ap_dict)
    # print("*"*100)
    print("mAP결과:", answer['map'])

    w = csv.writer(open("eval.csv", "w"))
    w.writerow(["Class", "Average Precision"])
    for key, val in ap_dict.items():
        w.writerow([key, val])


