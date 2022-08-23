import tensorflow as tf
from model.model_builder import ModelBuilder
from utils.model_post_processing import merge_post_process
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils.priors import *        
import argparse 

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
                    default='0818/_0818_efficient_lite_v0_wider+ffdb-test_e100_lr0.001_b32_best_loss.h5')
parser.add_argument("--num_classes",         type=int,    help="Set num classes for model and post-processing",
                    default=2)  
parser.add_argument("--image_size",          type=tuple,  help="Set image size for priors and post-processing",
                    default=(300, 300))
parser.add_argument("--gpu_num",             type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)
parser.add_argument("--frozen_dir",          type=str,    help="Path to save frozen graph transformation result",
                    default='./checkpoints/converted_frozen_graph/')
parser.add_argument("--frozen_name",         type=str,    help="Frozen graph file name to save",
                    default='frozen_graph')
parser.add_argument("--include_postprocess",   help="Frozen graph file name to save",
                    action='store_true')
            
args = parser.parse_args()

if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)
    tf.config.run_functions_eagerly(True)
    
    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):

        spec_list = convert_spec_list()
        priors = create_priors_boxes(specs=spec_list, image_size=args.image_size[0], clamp=True)
        target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

        model = ModelBuilder(image_size=args.image_size,
                                    num_classes=args.num_classes, include_preprocessing=args.include_postprocess).build_model(args.backbone_name)

        model.load_weights(args.checkpoint_dir + args.model_weights, by_name=True)

        if args.include_postprocess:
            detection_output = merge_post_process(detections=model.output,
                                                  target_transform=target_transform,
                                                  confidence_threshold=0.5,
                                                  classes=args.num_classes)
            model = Model(inputs=model.input, outputs=detection_output)

        model.summary()

        #path of the directory where you want to save your model
        frozen_out_path = args.frozen_dir
        # name of the .pb file
        frozen_graph_filename = args.frozen_name
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        
        print("Frozen model inputs: {0}".format(frozen_func.inputs))
        print("Frozen model outputs: {0}".format(frozen_func.outputs))
        
        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pb",
                        as_text=False)
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pbtxt",
                        as_text=True)