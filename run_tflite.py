import tensorflow as tf
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input

TFLITE_FILE_PATH = './checkpoints/_tf_lite_model.tflite'


interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
#interpreter.resize_tensor_input(0, [1, 300, 300, 3], strict=True)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()

    # 텐서 변환
    input = tf.convert_to_tensor(frame)
    #input = tf.image.decode_jpeg(input, channels=3)
    # 이미지 리사이징
    input = tf.image.resize(input, [512, 512])
    input = preprocess_input(input, mode='torch')
    input = tf.expand_dims(input, axis=0)


    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
