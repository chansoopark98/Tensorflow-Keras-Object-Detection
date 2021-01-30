import tensorflow_datasets as tfds
from utils.priors import *
from preprocessing import prepare_dataset


from tqdm import tqdm
from pprint import pprint
import csv


DATASET_DIR = 'datasets/'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/0129_main.h5'
EVAL_MODE = 'pascal' # 'pascal' or 'kitti'


if EVAL_MODE == 'pascal':
    from model.pascal_main import ssd
    from utils.pascal.voc_evaluation import eval_detection
    from utils.pascal.pascal_post_processing import post_process


    test_data = tfds.load('voc', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    classes = 21
    difficult_name = 'is_difficult'
    label_name = 'label'

elif EVAL_MODE == 'kitti':
    from model.kitti_main import ssd
    from utils.kitti.kitti_evaluation import eval_detection
    from utils.kitti.kitti_post_processing import post_process

    test_data = tfds.load('kitti', data_dir=DATASET_DIR, split='test')
    number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
    # number_test = 4952
    print("테스트 데이터 개수:", number_test)
    classes = 8
    difficult_name = 'truncated'
    label_name = 'type'


with open(EVAL_MODE+'_labels.txt') as f:
    CLASSES = f.read().splitlines()

iou_threshold = 0.5
center_variance = 0.1
size_variance = 0.2

specs = [
                Spec(38, 8, BoxSizes(30, 60), [2]),
                Spec(19, 16, BoxSizes(60, 111), [2, 3]),
                Spec(10, 32, BoxSizes(111, 162), [2, 3]),
                Spec(5, 64, BoxSizes(162, 213), [2, 3]),
                Spec(3, 100, BoxSizes(213, 264), [2]),
                Spec(1, 300, BoxSizes(264, 315), [2])
        ]


priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

# instantiate the datasets
validation_dataset = prepare_dataset(test_data, IMAGE_SIZE, BATCH_SIZE, target_transform, classes, train=False)
model = ssd(MODEL_NAME, pretrained=False)
model.load_weights(checkpoint_filepath)
validation_steps = number_test // BATCH_SIZE + 1


test_bboxes = []
test_labels = []
test_difficults = []
for sample in test_data:
    label = sample['objects'][label_name].numpy()
    bbox = sample['objects']['bbox'].numpy()[:,[1, 0, 3, 2]]
    #is_difficult = sample['objects'][difficult_name].numpy()
    is_difficult = None

    test_bboxes.append(bbox)
    test_labels.append(label)
    test_difficults.append(is_difficult)

print("Evaluating..")
pred_bboxes = []
pred_labels = []
pred_scores = []
for batch in tqdm(validation_dataset, total=validation_steps):
    pred = model.predict_on_batch(batch)
    predictions = post_process(pred, target_transform)
    for prediction in predictions:
        boxes, scores, labels = prediction
        pred_bboxes.append(boxes)
        pred_labels.append(labels.astype(int) - 1)
        pred_scores.append(scores)

answer = eval_detection(pred_bboxes=pred_bboxes,
                        pred_labels=pred_labels,
                        pred_scores=pred_scores,
                        gt_bboxes=test_bboxes,
                        gt_labels=test_labels,
                        gt_difficults=test_difficults,
                        use_07_metric=True)
print("*"*100)
print("Average Precisions")
ap_dict = dict(zip(CLASSES, answer['ap']))
pprint(ap_dict)
print("*"*100)
print("Mean Average Precision:", answer['map'])

w = csv.writer(open("eval.csv", "w"))
w.writerow(["Class", "Average Precision"])
for key, val in ap_dict.items():
    w.writerow([key, val])