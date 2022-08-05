import tensorflow as tf
from model.model_builder import ModelBuilder
from utils.model_post_processing import merge_post_process
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils.priors import *        
import argparse 
# quantize_uint8
# tensorflowjs_converter ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 
# tensorflowjs_converter ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 
# tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Identity' ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/converted_tfjs/

# Set Distribute training (When use Multi gpu)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num",          type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)


args = parser.parse_args()

if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    
    tf.config.set_soft_device_placement(True)

    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):

        IMAGE_SIZE = (300, 300)
        num_classes = 21
        checkpoints = './checkpoints/0805/_0805_efficient_lite_v0_b16_e100_single_gpu_best_loss.h5'

        spec_list = convert_spec_list()
        priors = create_priors_boxes(specs=spec_list, image_size=IMAGE_SIZE[0], clamp=True)
        target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

        model = ModelBuilder(image_size=IMAGE_SIZE,
                                    num_classes=num_classes).build_model('efficient_lite_v0')
        model.load_weights(checkpoints, by_name=True)
        detection_output = merge_post_process(detections=model.output, target_transform=target_transform, confidence_threshold=0.5)
        model = Model(inputs=model.input, outputs=detection_output)

        model.summary()

        #path of the directory where you want to save your model
        frozen_out_path = './checkpoints/new_tfjs_frozen'
        # name of the .pb file
        frozen_graph_filename = "frozen_graph"
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pb",
                        as_text=False)
        # Save its text representation
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pbtxt",
                        as_text=True)

