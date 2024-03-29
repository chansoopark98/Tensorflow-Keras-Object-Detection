import tensorflow as tf
from model.model_builder import ModelBuilder
from utils.model_post_processing import merge_post_process
from utils.priors import *        
import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_name",       type=str,    help="Pretrained backbone name\
                                                            |   model_name    : description | \
                                                            [ 1. mobilenetv2       : MobileNetV2 ]\
                                                            [ 2. mobilenetv3s      : MobileNetV3-Small ] \
                                                            [ 3. mobilenetv3l      : MobileNetV3-Large ] \
                                                            [ 4. efficient_lite_v0 : EfficientNet-Lite-B0 ]\
                                                            [ 5. efficientnetv2b0  : EfficientNet-V2-B0 ]\
                                                            [ 6. efficientnetv2b3  : EfficientNet-V2-B3 ]",
                    default='efficient_lite_v0')
parser.add_argument("--checkpoint_dir",      type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--model_weights",       type=str,    help="Saved model weights directory",
                    default='your_model_weights.h5')
parser.add_argument("--num_classes",         type=int,    help="Set num classes for model and post-processing",
                    default=21)  
parser.add_argument("--image_size",          type=tuple,  help="Set image size for priors and post-processing",
                    default=(300, 300))
parser.add_argument("--gpu_num",             type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)
parser.add_argument("--export_dir",          type=str,    help="Path to save frozen graph transformation result",
                    default='./checkpoints/tflite_converted/')
parser.add_argument("--tflite_name",         type=str,    help="TFlite file name to save",
                    default='tflite.tflite')
parser.add_argument("--include_postprocess", help="Frozen graph file name to save",
                    action='store_true')
parser.add_argument("--test",                help="Frozen graph file name to save",
                    action='store_true')
            
args = parser.parse_args()

if __name__ == '__main__':
    

    os.makedirs(args.export_dir, exist_ok=True)
    # args.test= True
    if args.test:
        with tf.device('cpu'):

            spec_list = convert_spec_list()
            priors = create_priors_boxes(specs=spec_list, image_size=args.image_size[0], clamp=True)
            target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=args.export_dir + args.tflite_name)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            print(input_details)
            output_details = interpreter.get_output_details()
            print(output_details)

            # Test the model on random input data.
            print('get test model')
            input_shape = input_details[0]['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            print(input_data.shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            print('get output data')
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)
            detection_output = merge_post_process(detections=output_data,
                                    target_transform=target_transform,
                                    confidence_threshold=0.5,
                                    classes=args.num_classes)
            print(detection_output)

    else:
        gpu_number = '/device:GPU:' + str(args.gpu_num)
        with tf.device(gpu_number):
            
            # tf.config.set_soft_device_placement(True)
            tf.config.run_functions_eagerly(True)

            

            model = ModelBuilder(image_size=args.image_size,
                                        num_classes=args.num_classes, include_preprocessing=args.include_postprocess).build_model(args.backbone_name)

            model.load_weights(args.checkpoint_dir + args.model_weights, by_name=True)

            model.summary()

            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()

            # Save the model.
            with open(args.export_dir + args.tflite_name, 'wb') as f:
                f.write(tflite_model)