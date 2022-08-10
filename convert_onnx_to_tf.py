from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from onnx_tf.backend import prepare as prepare_onnx_model
import tensorflow as tf
import argparse
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_dir",     type=str,    help="Path where ONNX models are stored (.onnx)",
                    default='your_onnx_model_dir.onnx')
parser.add_argument("--output_dir",     type=str,    help="Path to save the converted model with tensorflow",
                    default='onnx2tf_converted/')
parser.add_argument("--gpu_num",          type=int,    help="Specify the GPU to perform the conversion on",
                    default=0)
args = parser.parse_args()

if __name__ == '__main__':
    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):
        """
            ONNX -> Tensorflow saved model
        """
        # Load the ONNX model and convert it to a tensorflow saved model.
        onnx_model = onnx.load(args.onnx_dir)
        onnx2tf_model = prepare_onnx_model(onnx_model)
        onnx2tf_model.export_graph(args.output_dir + 'onnx2tf_model')

        """
            Tensorflow savedf model -> Tensorflow frozen graph
        """
        # Load the saved tensorflow saved model.
        model = tf.saved_model.load(args.output_dir + 'onnx2tf_model')

        # Convert to frozen graph.
        frozen_out_path = args.output_dir + 'frozen_graph_result'
        # Set name of the frozen graph (.pb) file
        frozen_graph_filename = 'frozen_graph'

        full_model = tf.function(lambda x: model(images=x))  # full model                                                  
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.signatures['serving_default'].inputs[0].shape.as_list(),
                          model.signatures['serving_default'].inputs[0].dtype.name))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("Frozen model inputs: {0}".format(frozen_func.inputs))
        print("Frozen model outputs: {0}".format(frozen_func.outputs))

        # Save frozen graph
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pb",
                        as_text=False)
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pbtxt",
                        as_text=True)