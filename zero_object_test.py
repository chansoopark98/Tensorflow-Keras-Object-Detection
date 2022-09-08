import tensorflow as tf
from utils.priors import *
import tensorflow_datasets as tfds

test_data = tfds.load('voc_zero', data_dir='./datasets/', split='train')
# test_data = tfds.load('display_detection', data_dir='./datasets/', split='train')

spec_list = convert_spec_list()
priors = create_priors_boxes(specs=spec_list, image_size=300, clamp=True)
target_transform = MatchingPriors(priors, center_variance, size_variance, iou_threshold)

num_classes = 4
image_size = (300, 300)

for sample in test_data.take(1000):
    
    image = sample['image'].numpy()
    bbox = sample['bbox'].numpy()
    labels = sample['label'].numpy() + 1

    x_min = bbox[:,1]
    y_min = bbox[:,0]
    x_max = bbox[:,3]
    y_max = bbox[:,2]
    bbox = tf.stack([x_min, y_min, x_max, y_max], axis=1)

    locations, labels = target_transform(tf.cast(bbox, tf.float32), labels)
    # tf.where(labels==0, 1)
    objectness = tf.cast(tf.where(labels>0, 1, 0), tf.float32)
    objectness = tf.expand_dims(objectness , axis=-1)

    print(locations)
    locations *= objectness
    
    
    locations = tf.clip_by_value(locations, 0, 1000)
    locations = tf.where(tf.math.is_nan(locations), 0.5, locations)
    print(locations)

    pos_mask = labels > 0
    locations = tf.reshape(tf.boolean_mask(locations, pos_mask), [-1, 4])

    

    locations = tf.where(tf.shape(locations)[0] == 0, tf.zeros([1, 4]), locations)
    print(locations)

    pred_locs = tf.ones((1, 4))

    diff = pred_locs-locations
    abs_diff = tf.abs(diff)
    sigma = 1.0
    clipping = tf.less(abs_diff, 1/(sigma**2))
    choose_value = tf.where(clipping, 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))
    print(choose_value)
    

            

    # labels = tf.one_hot(labels, num_classes, axis=1, dtype=tf.float32)
    # # print(bbox, locations[0], labels[0])
    # targets = tf.concat([labels, locations, objectness], axis=1)
    # resized_img = tf.image.resize(image, image_size)
    # # print(targets)
    
    
    