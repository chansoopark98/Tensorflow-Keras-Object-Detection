import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils.priors import *
from utils.model_post_processing import post_process
from model.model_builder import ModelBuilder
from utils.misc import draw_bounding, CLASSES, COCO_CLASSES, TEST_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_name",       type=str,   help="Pretrained backbone name",
                    default='efficient_lite_v0')
parser.add_argument("--num_classes",          type=int,   help="Number of classes in the pretrained model",
                    default=2)
parser.add_argument("--train_dataset_type",  type=str,   help="Train dataset type",
                    default='voc')
parser.add_argument("--image_size",          type=tuple, help="Model image size (input resolution)",
                    default=(300, 300))
parser.add_argument("--threshold",           type=float, help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,   help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,   help="Saved model weights directory",
                    default='your_model_weights.h5')
parser.add_argument("--gpu_num",             type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)
args = parser.parse_args()

if __name__ == '__main__':
    # Set num classes and label list by args.train_dataset_type
    tf.config.set_soft_device_placement(True)

    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):
        label_list = TEST_CLASSES

        # Set target transforms
        spec_list = convert_spec_list()
        priors = create_priors_boxes(specs=spec_list, image_size=args.image_size[0], clamp=True)
        target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

        model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model(args.backbone_name)
        model.load_weights(args.checkpoint_dir + args.weight_name)
        model.summary()

        # Camera
        frame_width = 480
        frame_height = 640
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        while cv2.waitKey(33) < 0:
            ret, frame = capture.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = tf.image.resize(img, size=args.image_size,
                    method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32)
            img /= 255.
            
            img = tf.expand_dims(img, axis=0)

            pred = model.predict(img)

            predictions = post_process(pred, target_transform, classes=args.num_classes, confidence_threshold=args.threshold)
            
            pred_boxes, pred_scores, pred_labels = predictions[0]

            if pred_boxes.size > 0:
                draw_bounding(frame, pred_boxes,  labels=pred_labels,  scores=pred_scores, img_size=frame.shape[:2], label_list=label_list)
            

            cv2.imshow("VideoFrame", frame)

        capture.release()
        cv2.destroyAllWindows()