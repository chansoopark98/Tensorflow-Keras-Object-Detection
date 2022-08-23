import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",     type=str,   help="저장된 wider_face dataset 경로 (tfds)", default='./datasets/')
parser.add_argument("--save_dir",     type=str,   help="변환 파일 저장 경로", default='./data_generate/data/wider_face/')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

image_path = args.save_dir + 'image/'
bbox_path = args.save_dir + 'bbox/'
label_path = args.save_dir + 'label/'

os.makedirs(image_path, exist_ok=True)
os.makedirs(bbox_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

train_data = tfds.load('wider_face', data_dir=args.dataset_dir, split='train')
valid_data = tfds.load('wider_face', data_dir=args.dataset_dir, split='validation')

wider_face = train_data.concatenate(valid_data)

wider_face = wider_face.filter(lambda x: tf.reduce_all(tf.not_equal(tf.size(x['faces']['bbox']), 0)))

number_wider_face = wider_face.reduce(0, lambda x, _: x + 1).numpy()

file_name = 0

for sample in wider_face.take(number_wider_face):
    file_name += 1
    print(file_name)
    image = sample['image'].numpy()
    boxes = sample['faces']['bbox'].numpy() 
    labels = tf.zeros(tf.shape(boxes)[0]).numpy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bbox_list = boxes.tolist()
    label_list = labels.tolist()

    cv2.imwrite(image_path + str(file_name) +'_.png', image)

    with open(bbox_path + str(file_name) +'_.txt', "w") as file:
        for i in range(len(bbox_list)):
            file.writelines(str(bbox_list[i]) + '\n')

    with open(label_path + str(file_name) +'_.txt', "w") as file:
        for i in range(len(label_list)):
            file.writelines(str(label_list[i]) + '\n')
                                
                    
        
        