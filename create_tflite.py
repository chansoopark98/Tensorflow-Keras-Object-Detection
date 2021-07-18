import tensorflow as tf

# model = tf.keras.models.load_model('./checkpoints/tflite/keras_model.h5')
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
# converter.experimental_new_converter =True
#
# tflite_model = converter.convert()
#
#
#



model =  tf.keras.models.load_model('./checkpoints/tflite/keras_model.h5',compile=False)#

model.summary()

model.save('./checkpoints/tflite/test.h5', save_format='tf')#.pb모델로 저장(폴더)

##pb.형식에서 tflite
converter = tf.lite.TFLiteConverter.from_saved_model('test')
tflite_model = converter.convert()

# Save the model.
with open('./checkpoints/tflite/'+'tf_lite_model.tflite', 'wb') as f:
    f.write(tflite_model)

