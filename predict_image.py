import os
import glob
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.priors import *
from utils.model_post_processing import post_process, merge_post_process
from model.model_builder import ModelBuilder
from utils.misc import draw_bounding, CLASSES, COCO_CLASSES, TEST_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_name",       type=str,    help="Pretrained backbone name",
                    default='efficient_lite_v0')
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=1)
parser.add_argument("--num_classes",         type=int,    help="Number of classes",
                    default=4)
parser.add_argument("--image_dir",           type=str,    help="Image directory",
                    default='./inputs/')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(300, 300))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='0811/_0811_efficient_lite_v0_new_display_detection_e100_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    image_list = os.path.join(args.image_dir, '*.jpg')
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)
    
    # Set target transforms
    spec_list = convert_spec_list()
    priors = create_priors_boxes(specs=spec_list, image_size=args.image_size[0], clamp=True)
    target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model(args.backbone_name)
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    for i in range(len(image_list)):
        frame = cv2.imread(image_list[i])

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(img, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)

        predictions = post_process(pred, target_transform, classes=args.num_classes, confidence_threshold=args.threshold, iou_threshold=0.5, top_k=100)

        pred_boxes, pred_scores, pred_labels = predictions[0]
        

        if pred_boxes.size > 0:
            draw_bounding(frame, pred_boxes,  labels=pred_labels,  scores=pred_scores, img_size=frame.shape[:2], label_list=CLASSES)
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tf.keras.preprocessing.image.save_img(result_dir + str(i)+'_.png', frame)

    cv2.destroyAllWindows()