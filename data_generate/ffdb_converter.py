from re import L
import numpy as np
import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",     type=str,   help="FFDB dataset dir", default='./data_generate/data/FFDB/')

args = parser.parse_args()


if __name__ == '__main__':

    fddb_labels = args.dataset_dir + '/FDDB-folds/'
    save_path = args.dataset_dir + '/results/'
    image_path = save_path + 'image/'
    bbox_path = save_path + 'bbox/'
    label_path = save_path + 'label/'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    label_lists = os.path.join(fddb_labels + '*-ellipseList.txt',)
    label_lists = glob.glob(label_lists)

    for label_file in label_lists:
        print(label_file)

        with open(label_file, "r") as file:
            line = file.readlines()

            while True:
                if len(line) == 0:
                    break

                

                file_name = line.pop(0)
                file_name = file_name.replace('\n', '')
                save_name = file_name.replace('/', '_')

                image = cv2.imread(args.dataset_dir + file_name + '.jpg')
                image_h, image_w, _ = image.shape

                samples = line.pop(0)
                samples = int(samples.replace('\n', ''))
                
                bbox_list = []
                label_list = []

                for sample_box in range(samples):
                    sample_box_line = line.pop(0)
                    sample_box_line = sample_box_line.replace('\n', '')
                    sample_box_line = sample_box_line.split(' ')
                    sample_box_line = list(filter(None, sample_box_line))
                    float_box = [float(str_box) for str_box in sample_box_line]

                    major, minor, _, x, y, _ = float_box
                    w = minor * 2
                    h = major * 2
                    x_min= int((x-w/2)) / image_w
                    y_min= int((y-h/2)) / image_h
                    x_max= int((x+w/2)) / image_w
                    y_max= int((y+h/2)) / image_h

                            
                    # clamp
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)

                    x_max = min(x_max, 1)
                    y_max = min(y_max, 1)
                    
                    voc_box = [y_min, x_min, y_max, x_max]

                    bbox_list.append(voc_box)
                    label_list.append(0)


                    
                cv2.imwrite(image_path + save_name +'_.png', image)

                with open(bbox_path + save_name +'_.txt', "w") as file:
                    for i in range(len(bbox_list)):
                        file.writelines(str(bbox_list[i]) + '\n')

                with open(label_path + save_name +'_.txt', "w") as file:
                    for i in range(len(label_list)):
                        file.writelines(str(label_list[i]) + '\n')
                                
                    

            