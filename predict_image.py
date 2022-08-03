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
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Number of classes", default=21)
parser.add_argument("--train_dataset_type",     type=str,
                    help="Train dataset type", default='voc')
parser.add_argument("--image_dir",    type=str,
                    help="Image directory", default='/home/park/park/Tensorflow-Keras-Realtime-Segmentation/data_augmentation/raw_data/bg/')
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(300, 300))
parser.add_argument("--threshold",     type=float,
                    help="Post processing confidence threshold", default=0.5)
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='0803/_0803_efficientv2b3_voc_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    image_list = os.path.join(args.image_dir, '*.jpg')
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    # Set num classes and label list by args.train_dataset_type

    

    
    # Set target transforms
    spec_list = convert_spec_list()
    priors = create_priors_boxes(specs=spec_list, image_size=args.image_size[0], clamp=True)
    target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

    model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model('efficientv2b3')
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    for i in range(len(image_list)):
        frame = cv2.imread(image_list[i])

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(img, tf.float32)
        img = preprocess_input(x=img, mode='torch')
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)

        # predictions = post_process(pred, target_transform, classes=args.num_classes, confidence_threshold=args.threshold, iou_threshold=0.5, top_k=100)

        # pred_boxes, pred_scores, pred_labels = predictions[0]
        
        
        predictions = merge_post_process(pred, target_transform, classes=args.num_classes, confidence_threshold=args.threshold, iou_threshold=0.5, top_k=100)
        output = predictions.numpy()
        print('predictions', predictions)

        pred_boxes = []
        pred_scores = []
        pred_labels = []

        if output.size > 0:
            print(output.size)
            print(output)
            for preds in output:
                *boxes, scores, labels = preds
                print(boxes)
                pred_boxes.append(boxes)
                pred_scores.append(scores)
                pred_labels.append(labels)

            pred_boxes = np.array(pred_boxes)
            pred_scores = np.array(pred_scores)
            pred_labels = np.array(pred_labels)

            draw_bounding(frame, pred_boxes,  labels=pred_labels, scores=pred_scores, img_size=frame.shape[:2], label_list=CLASSES)

        tf.keras.preprocessing.image.save_img(result_dir + str(i)+'_.png', frame)

    cv2.destroyAllWindows()