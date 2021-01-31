from utils.priors import *
from model.pascal_main import ssd
from utils.pascal_post_processing import post_process

import cv2

from collections import namedtuple

DATASET_DIR = 'datasets'
IMAGE_SIZE = [300, 300]
BATCH_SIZE = 16
MODEL_NAME = 'B0'
checkpoint_filepath = './checkpoints/1111_b0_ep400.h5'
INPUT_DIR = './inputs'
OUTPUT_DIR = './outputs'

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

model = ssd(MODEL_NAME, pretrained=False)
model.summary()

model.load_weights(checkpoint_filepath)




def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])


Label = namedtuple('Label', ['name', 'color'])


def color_map(index):
    label_defs = [
        Label('aeroplane', rgb2bgr((0, 0, 0))),
        Label('bicycle', rgb2bgr((111, 74, 0))),
        Label('bird', rgb2bgr((81, 0, 81))),
        Label('boat', rgb2bgr((128, 64, 128))),
        Label('bottle', rgb2bgr((244, 35, 232))),
        Label('bus', rgb2bgr((230, 150, 140))),
        Label('car', rgb2bgr((70, 70, 70))), # 7
        Label('cat', rgb2bgr((102, 102, 156))),
        Label('chair', rgb2bgr((190, 153, 153))),
        Label('cow', rgb2bgr((150, 120, 90))),
        Label('diningtable', rgb2bgr((153, 153, 153))),
        Label('dog', rgb2bgr((250, 170, 30))),
        Label('horse', rgb2bgr((220, 220, 0))),
        Label('motorbike', rgb2bgr((107, 142, 35))),
        Label('person', rgb2bgr((52, 151, 52))), # 15
        Label('pottedplant', rgb2bgr((70, 130, 180))),
        Label('sheep', rgb2bgr((220, 20, 60))),
        Label('sofa', rgb2bgr((0, 0, 142))),
        Label('train', rgb2bgr((0, 0, 230))),
        Label('tvmonitor', rgb2bgr((119, 11, 32)))]
    return label_defs[index]


def draw_bounding(img, bboxes, labels, img_size):
    # resizing 작업
    if np.max(bboxes) < 10:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_size[1]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_size[0]

    for i, bbox in enumerate(bboxes):
        xmin = tf.cast(bbox[0], dtype=tf.int32)
        ymin = tf.cast(bbox[1], dtype=tf.int32)
        xmax = tf.cast(bbox[2], dtype=tf.int32)
        ymax = tf.cast(bbox[3], dtype=tf.int32)
        img_box = np.copy(img)
        _, color = color_map(int(labels[i] - 1))
        cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_box, CLASSES[int(labels[i] - 1)], (xmin + 5, ymin - 5), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        alpha = 0.8
        cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)



from keras.applications.imagenet_utils import preprocess_input

videoFilePath = './dumping2.mp4'
cap = cv2.VideoCapture(videoFilePath)



while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:

        img = np.expand_dims(frame, axis=0)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.image.resize(img, [300,300])
        img = preprocess_input(img, mode='torch')




        pred = model.predict(img)
        predictions = post_process(pred, target_transform, confidence_threshold=0.3)

        pred_boxes, pred_scores, pred_labels = predictions[0]
        # pred_labels  = [x for x in pred_labels if x==float(7) or x==float(15)]
        # print(pred_scores)
        # print(pred_labels)

        if pred_boxes.size > 0:
            draw_bounding(frame, pred_boxes, labels=pred_labels, img_size=frame.shape[:2])



        cv2.imshow('video',frame)
        cv2.waitKey()
        print(ret)


cap.release()
cv2.destroyAllWindows()


