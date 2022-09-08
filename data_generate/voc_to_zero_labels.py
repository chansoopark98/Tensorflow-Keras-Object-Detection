import cv2
import glob
import os

RGB_PATH = './datasets/voc_image/'
SAVE_PATH = './datasets/voc_zero_raw/'
SAVE_RGB_PATH = SAVE_PATH + 'rgb/'
SAVE_LABEL_PATH = SAVE_PATH + 'label/'
SAVE_BBOX_PATH = SAVE_PATH + 'bbox/'

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_RGB_PATH, exist_ok=True)
os.makedirs(SAVE_LABEL_PATH, exist_ok=True)
os.makedirs(SAVE_BBOX_PATH, exist_ok=True)

rgb_list = glob.glob(os.path.join(RGB_PATH+'*.jpg'))

for image_path in rgb_list:
    image = cv2.imread(image_path)
    prefix = image_path.split('/')[3].split('.')[0]
    print(prefix)
    cv2.imwrite(SAVE_RGB_PATH + prefix + '.png', image)

    bbox = [0, 0, 0, 0]
    labels = -1
    with open(SAVE_BBOX_PATH + prefix +'_.txt', "w") as file:
        
        file.writelines(str(bbox) + '\n')

    with open(SAVE_LABEL_PATH + prefix +'_.txt', "w") as file:
        
        file.writelines(str(labels) + '\n')
