import tensorflow_datasets as tfds
from utils.priors import *
from model.model_builder import ssd
from preprocessing import pascal_prepare_dataset
from preprocessing import coco_eval_dataset
from utils.model_post_processing import post_process  #
from utils.model_evaluation import eval_detection_voc
from tensorflow.keras.utils import plot_model
from calc_flops import get_flops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from pprint import pprint
import csv
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0310.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='coco')
parser.add_argument("--calc_flops",  type=str,   help="모델 FLOPS 계산", default=False)
args = parser.parse_args()

BATCH_SIZE = 1
IMAGE_SIZE = [args.image_size, args.image_size]
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CALC_FLOPS = args.calc_flops

os.makedirs(DATASET_DIR, exist_ok=True)



if TRAIN_MODE == 'voc':
    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    print("테스트 데이터 개수:", number_test)
    CLASSES_NUM = 21
    BATCH_SIZE = 32
    with open('./pascal_labels.txt') as f:
        CLASSES = f.read().splitlines()

else:
    test_data, test_info = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation', with_info=True)

    # TODO TEST 데이터셋 다운로드 후 재구축
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
    test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()

    print("테스트 데이터 개수:", number_test)
    CLASSES_NUM = 81
    with open('./coco_labels.txt') as f:
        CLASSES = f.read().splitlines()

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
    Spec(48, 8, BoxSizes(38, 77), [2]),  # 0.1
    Spec(24, 16, BoxSizes(77, 142), [2, 3]),  # 0.2
    Spec(12, 32, BoxSizes(142, 207), [2, 3]),  # 0.37
    Spec(6, 64, BoxSizes(207, 273), [2, 3]),  # 0.54
    Spec(3, 128, BoxSizes(273, 337), [2]),  # 0.71
    Spec(1, 384, BoxSizes(337, 403), [2])  # 0.88 , max 1.05
]

priors = create_priors_boxes(specs, IMAGE_SIZE[0])
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

# instantiate the datasets
validation_dataset = pascal_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                            target_transform, TRAIN_MODE, train=False)
coco_dataset = coco_eval_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                    target_transform, TRAIN_MODE, train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = ssd(TRAIN_MODE, MODEL_NAME, pretrained=False)

print("모델 가중치 로드...")
model.load_weights(CHECKPOINT_DIR)
model.summary()

if CALC_FLOPS:
    plot_model(model,'model_plot.png',show_shapes=False)
    flops = get_flops(model, BATCH_SIZE)
    print(f"FLOPS: {flops}")

test_steps = number_test // BATCH_SIZE + 1
print("테스트 배치 개수 : ", test_steps)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if TRAIN_MODE == 'coco':
    img_id = []
    cat_id = []
    pred_ids = []
    pred_labels = []
    pred_boxes = []
    pred_scores = []
    pred_list = []
    img_shapes = []

    # (img, img_shape , img_id, cat_id)
    # for x, img_shape, img_id, cat_id in tqdm(coco_dataset, total=test_steps):

    for sample in test_data:
        img_shapes.append(sample['image'].shape)
        img_id.append(np.int(sample['image/id'].numpy().astype('int32').item()))

    for x, pred_id in tqdm(coco_dataset, total=test_steps):
        pred = model.predict_on_batch(x)
        predictions = post_process(pred, target_transform, classes=CLASSES_NUM)
        pred_ids.append(pred_id.numpy().astype('int32').item())
        for boxes, scores, labels in predictions:
            pred_labels.append(np.round(labels.tolist(),2).astype('int32'))
            pred_boxes.append(np.round(boxes.tolist(),2).astype('float32'))
            pred_scores.append(np.round(scores.tolist(),2).astype('float32'))


    for index in range(len(pred_ids)):
        for i in range(len(pred_labels[index])):
            bbox = pred_boxes[index][i]
            xmin, ymin, xmax, ymax = bbox

            xmin = xmin * img_shapes[index][1]
            ymin = ymin * img_shapes[index][0]
            xmax = xmax * img_shapes[index][1]
            ymax = ymax * img_shapes[index][0]
            x = xmin
            y = ymin
            w = (xmax - xmin) + 1
            h = (ymax - ymin) + 1
            total_predictions = {"image_id": pred_ids[index],
                             "category_id": pred_labels[index][i],
                             "bbox": [x, y, w, h],
                             "score": pred_scores[index][i]
                             }

            pred_list.append(total_predictions)

    with open('datasets/coco_predictions.json', 'w') as f:
        json.dump(pred_list, f, ensure_ascii=False, indent="\t", cls=NumpyEncoder)

    annType = 'bbox'
    cocoGt = COCO('datasets/instances_val2017.json')
    cocoDt = cocoGt.loadRes('datasets/coco_predictions.json')
    imgIds = sorted(cocoGt.getImgIds())
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

else:
    test_difficults = []
    use_07_metric = True
    test_bboxes = []
    test_labels = []

    for sample in test_data:
        label = sample['objects']['label'].numpy()
        bbox = sample['objects']['bbox'].numpy()[:, [1, 0, 3, 2]]

        is_difficult = sample['objects']['is_difficult'].numpy()
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

