from model_configuration import ModelConfiguration
import tensorflow as tf
import argparse
import time

# LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model",      help="Convert to saved model format",
                    action='store_true')
parser.add_argument("--saved_model_path", type=str,   help="Saved model weight path",
                    default='pretrained_weights')

# Build with post processing
parser.add_argument("--build_postprocess",  help="Post processing build",
                    action='store_true')

# Training use pre-trained mode (voc, coco .. etc)
parser.add_argument("--transfer_learning",  help="Load the pre-trained weights and proceed with further training.",
                    action='store_true')

# Set Training Options
parser.add_argument("--model_prefix",     type=str,    help="Model name (logging weights name and tensorboard)",
                    default='efficient_lite_v0_wider+ffdb-test_e100_lr0.001_b32')
parser.add_argument("--backbone_name",    type=str,    help="Pretrained backbone name\
                                                            |   model_name    : description | \
                                                            [ 1. mobilenetv2       : MobileNetV2 ]\
                                                            [ 2. mobilenetv3s      : MobileNetV3-Small ] \
                                                            [ 3. mobilenetv3l      : MobileNetV3-Large ] \
                                                            [ 4. efficient_lite_v0 : EfficientNet-Lite-B0 ]\
                                                            [ 5. efficientv2b0  : EfficientNet-V2-B0 ]\
                                                            [ 6. efficientv2b3  : EfficientNet-V2-B3 ]",
                    default='efficient_lite_v0')
parser.add_argument("--batch_size",       type=int,    help="Batch size per each GPU",
                    default=32)
parser.add_argument("--epoch",            type=int,    help="Training epochs",
                    default=100)
parser.add_argument("--lr",               type=float,  help="Initial learning rate",
                    default=0.001)
parser.add_argument("--weight_decay",     type=float,  help="Set Weight Decay",
                    default=0.00001)
parser.add_argument("--image_size",       type=tuple,  help="Set model input size",
                    default=(300, 300))
parser.add_argument("--image_norm_type",  type=str,    help="Set RGB image nornalize format (tf or torch or no)\
                                                             [ 1. tf    : Rescaling RGB image -1 ~ 1 from imageNet ]\
                                                             [ 2. torch : Rescaling RGB image 0 ~ 1 from imageNet ]\
                                                             [ 3. else  : Rescaling RGB image 0 ~ 1 only divide 255 ]",
                    default='div')
parser.add_argument("--loss_type",        type=str,    help="Set the loss function for class classification.\
                                                            [ 1. ce     : Uses cross entropy with hard negative mining ]\
                                                            [ 2. focal  : Focal cross entropy ]",
                    default='ce')
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
parser.add_argument("--dataset_name",     type=str,    help="Set the dataset type. \
                                                             [ 1. voc : PASCAL VOC 07+12 dataset ] \
                                                             [ 2. coco : COCO2017 dataset ] \
                                                             [ 3. wider_face : Wider Face dataset ] \
                                                             [ 4. custom : Custom TFDS ]",
                    default='wider_face')
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