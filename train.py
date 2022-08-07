from model_configuration import ModelConfiguration
import tensorflow as tf
import argparse
import time

# 1. sudo apt-get install libtcmalloc-minimal4
# 2. check dir ! 
# dpkg -L libtcmalloc-minimal4
# 3. LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

# https://npmmirror.com/package/@tensorflow/tfjs-tflite
# https://groups.google.com/a/tensorflow.org/g/tflite/c/Yt6-eSMn_bg -> tflite 사용법 및 -127 ~ 128 정규화
# https://codesandbox.io/s/kwmq4v263?file=/src/index.js 텐서플로js sample files

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model",  help="SavedModel.pb 변환", action='store_true')
parser.add_argument("--saved_model_path", type=str,   help="Saved model weight path",
                    default='./checkpoints/0805/_0805_efficientv2b0_b32_e200_single_gpu_best_loss.h5')

# Build with post processing
parser.add_argument("--build_postprocess",  help="Post processing build", action='store_true')

# Set Training Options
parser.add_argument("--model_prefix",     type=str,    help="Model name",
                    default='efficient_lite_v0_lr0.002_b32_e300_single_gpu_bigger_adam_base-128')
parser.add_argument("--backbone_name",    type=str,    help="Pretrained backbone name",
                    default='efficient_lite_v0')
parser.add_argument("--batch_size",       type=int,    help="Batch size per each GPU",
                    default=32)
parser.add_argument("--epoch",            type=int,    help="Training epochs",
                    default=300)
parser.add_argument("--lr",               type=float,  help="Initial learning rate",
                    default=0.002)
parser.add_argument("--weight_decay",     type=float,  help="Set Weight Decay",
                    default=0.00001)
parser.add_argument("--image_size",       type=tuple,  help="Set model input size",
                    default=(300, 300))
parser.add_argument("--image_norm_type",  type=str,    help="Set RGB image nornalize format (tf or torch or no)",
                    default='torch')
parser.add_argument("--optimizer",        type=str,    help="Set optimizer",
                    default='adam')
parser.add_argument("--use_weightDecay",  type=bool,   help="Whether to use weightDecay",
                    default=False)
parser.add_argument("--mixed_precision",  type=bool,   help="Whether to use mixed_precision",
                    default=True)
parser.add_argument("--model_name",       type=str,    help="Set the model name to save",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))

# Set directory path (Dataset,  Dataset_type, Chekcpoints, Tensorboard)
parser.add_argument("--dataset_dir",      type=str,    help="Set the dataset download directory",
                    default='./datasets/')
parser.add_argument("--dataset_name",     type=str,    help="Set the dataset type (cityscapes, custom etc..)",
                    default='voc')
parser.add_argument("--checkpoint_dir",   type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,    help="Set tensorboard storage path",
                    default='tensorboard/')

# Set Distribute training (When use Single gpu)
parser.add_argument("--gpu_num",          type=int,    help="Set GPU number to use(When without distribute training)",
                    default=1)

# Set Distribute training (When use Multi gpu)
parser.add_argument("--multi_gpu",  help="Set up distributed learning mode", action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    if args.saved_model:
        model = ModelConfiguration(args=args)
        model.saved_model()

    else:
        if args.multi_gpu == False:
            tf.config.set_soft_device_placement(True)

            gpu_number = '/device:GPU:' + str(args.gpu_num)
            with tf.device(gpu_number):
                model = ModelConfiguration(args=args)
                model.train()

        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = ModelConfiguration(args=args, mirrored_strategy=mirrored_strategy)
                model.train()