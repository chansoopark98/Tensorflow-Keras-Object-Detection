import tensorflow_datasets as tfds
from utils.priors import *
from model.model_builder import model_build
from preprocessing import pascal_prepare_dataset
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
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
# parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0410_81.9%_b1_/0410.h5')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/coco_0501.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
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

    validation_dataset = pascal_prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                                target_transform, TRAIN_MODE, train=False)

else:
    if EVAL_TESTDEV:
        test_data, test_info = tfds.load('coco/2017', data_dir=DATASET_DIR, split='validation', with_info=True)
        # test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['bbox']), 0)))
        # test_data = test_data.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['objects']['label']), 0)))

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

    CLASSES_NUM = 81
    coco_dataset = coco_eval_dataset(test_data, IMAGE_SIZE, BATCH_SIZE,
                                     target_transform, TRAIN_MODE, train=False)

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


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
        # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
        # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    if TRAIN_MODE == 'coco':
        img_id = []
        cat_id = []
        pred_ids = []
        pred_labels = []
        pred_boxes = []
        pred_scores = []
        pred_list = []
        img_shapes = []

        removed_classes = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
        # (img, img_shape , img_id, cat_id)
        # for x, img_shape, img_id, cat_id in tqdm(coco_dataset, total=test_steps):

        for sample in test_data:
            img_shapes.append(sample['image'].shape)
            img_id.append(np.int(sample['image/id'].numpy().astype('int32').item()))

        for x, pred_id in tqdm(coco_dataset, total=test_steps):
            pred = model.predict_on_batch(x)
            # predictions = post_process(pred, target_transform, confidence_threshold=0.001, top_k=200, iou_threshold=0.4, classes=CLASSES_NUM)
            predictions = post_process(pred, target_transform, confidence_threshold=0.05, classes=CLASSES_NUM)
            pred_ids.append(pred_id)

            for boxes, scores, labels in predictions:
                for i in range(len(labels)):
                    stack = 0
                    for j in range(len(removed_classes)):
                        if labels[i] >= removed_classes[j]:
                            # stack += 1
                            labels[i] += 1
                    # labels[i] += stack


                pred_labels.append(labels)
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
                x = round(float(xmin), 2)
                y = round(float(ymin), 2)
                w = round(float((xmax - xmin) + 1), 2)
                h = round(float((ymax - ymin) + 1), 2)
                total_predictions = {"image_id": int(pred_ids[index]),
                                 "category_id": int(pred_labels[index][i]),
                                 "bbox": [x, y, w, h],
                                 "score": float(pred_scores[index][i])
                                 }

                pred_list.append(total_predictions)

        with open('datasets/coco_predictions.json', 'w') as f:
            json.dump(pred_list, f, indent="\t", cls=NumpyEncoder)

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


