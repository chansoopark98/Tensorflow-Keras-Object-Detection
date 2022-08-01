from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import tensorflow as tf
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",        type=str,   help="raw image path", 
                    default='./data_generate/data/rgb/')
parser.add_argument("--mask_path",       type=str,   help="raw mask path",
                    default='./data_generate/data/mask/')
parser.add_argument("--obj_mask_path",   type=str,   help="raw obj mask path",
                    default='./data_generate/data/obj_mask/')
parser.add_argument("--label_map_path",  type=str,   help="CVAT's Segmentation Mask format labelmap.txt path",
                    default='./data_generate/data/labelmap.txt')
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result",
                    default='./data_generate/data/augment/')

args = parser.parse_args()

class MaskDataGenerator():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask segObj)
                >>>    rgb : RGB image.
                >>>    mask : segmentation mask.
                >>>    segObj : segmentation object mask.
                >>>    label_map : segmentation mask(label) information.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        self.OBJ_MASK_PATH = args.obj_mask_path
        self.LABEL_MAP_PATH = args.label_map_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_LABEL_PATH = self.OUTPUT_PATH + 'label/'
        self.OUT_BBOX_PATH = self.OUTPUT_PATH + 'bbox/'
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_LABEL_PATH, exist_ok=True)
        os.makedirs(self.OUT_BBOX_PATH, exist_ok=True)

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.obj_mask_list = glob.glob(os.path.join(self.OBJ_MASK_PATH+'*.png'))
        self.obj_mask_list = natsort.natsorted(self.obj_mask_list,reverse=True)
        
        # Check your data (RGB file samples = Mask file samples)
        self.check_image_len() 

        # Get label information from labelmap.txt (Segmentation mask 1.1 format)
        self.label_list = self.get_label_list()


    def get_label_list(self) -> list:
        """
            Get label information from labelmap.txt
        """
        label_list = []

        with open(self.LABEL_MAP_PATH, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx != 0:
                    split_line = line.split(':')

                    class_name = split_line[0]
                    string_rgb = split_line[1]

                    r, g, b = string_rgb.split(',')
                    r = int(r)
                    g = int(g)
                    b = int(b)

                    output = {'class_name': class_name,
                              'class_idx': idx-1,
                              'rgb': (r, g, b)}

                    label_list.append(output)
        _ = label_list.pop(0)
        return label_list


    def check_image_len(self):
        """
            Check rgb, mask, obj mask sample counts
        """
        rgb_len = len(self.rgb_list)
        mask_len = len(self.mask_list)
        obj_mask_len = len(self.obj_mask_list)

        if rgb_len != mask_len:
            raise Exception('RGB Image files : {0}, Mask Image files : {1}. Check your image and mask files '
                            .format(rgb_len, mask_len))
        
        if rgb_len != obj_mask_len:
            raise Exception('RGB Image files : {0}, Object Mask Image files : {1}. Check your image and obj mask files '
                            .format(rgb_len, obj_mask_len))


    def get_rgb_list(self) -> list:
        """
            return rgb list instance
        """
        return self.rgb_list

    def get_mask_list(self) -> list:
        """
            return mask list instance
        """
        return self.mask_list


    def get_obj_mask_list(self) -> list:
        """
            return obj mask list instance
        """
        return self.obj_mask_list


    def image_resize(self, rgb: np.ndarray, mask: np.ndarray,
                     obj_mask: np.ndarray, size=(1600, 900)):
        """
            Image resizing function    
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                obj_mask (np.ndarray) : (H,W,1) Image.
                size     (tuple)      : Image size to be adjusted.
        """
        resized_rgb = tf.image.resize(images=rgb, size=size, method=tf.image.ResizeMethod.BILINEAR)
        resized_rgb = resized_rgb.numpy().astype(np.uint8)

        resized_mask = tf.image.resize(images=mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_mask = resized_mask.numpy().astype(np.uint8)

        
        resized_obj_mask = tf.image.resize(images=obj_mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_obj_mask = resized_obj_mask.numpy().astype(np.uint8)

        return resized_rgb, resized_mask, resized_obj_mask
        
        
    
    def image_histogram_equalization(self, rgb: np.ndarray) -> np.ndarray:
        """
            Image histogram equalization function
            Args:
                rgb (np.ndarray) : (H,W,3) Image.    
        """
        
        split_r, split_g, split_b = cv2.split(rgb)

        hist_r = cv2.equalizeHist(split_r)
        hist_g = cv2.equalizeHist(split_g)
        hist_b = cv2.equalizeHist(split_b)

        equ = cv2.merge((hist_r, hist_g, hist_b))

        return equ

    def image_random_bluring(self, rgb: np.ndarray, gaussian_min: int = 7,
                             gaussian_max: int = 21) -> np.ndarray:
        """
            NxN random gaussianBlurring function   
            Args:
                rgb           (np.ndarray) : (H,W,3) Image.
                gaussian_min  (int)        : Random gaussian kernel minimum size
                gaussian_max  (int)        : Random gaussian kernel maximum size
        """
        k = random.randrange(gaussian_min, gaussian_max, 2)    
        rgb = cv2.GaussianBlur(rgb, (k,k), 0)

        return rgb


    def image_random_brightness(self, rgb: np.ndarray, scale_min: float = 0.5,
                                scale_max: float = 1.0, factor: int = 100) -> np.ndarray:
        """
            Random brightness function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                scale_min  (float)      : Minimum change in random blurring
                scale_max  (float)      : Maximum change in random blurring
                factor     (int)        : The maximum intensity of blurring,
                                          multiplied by a value between scale_min and scale_max.
        """
        random_val = np.random.uniform(low=scale_min, high=scale_max)
        factor *= random_val
        array = np.full(rgb.shape, (factor, factor, factor), dtype=np.uint8)

        return cv2.add(rgb, array)

    def image_random_translation(self, rgb: np.ndarray, mask: np.ndarray, obj_mask: np.ndarray,
                                 min_dx: int, min_dy: int,
                                 max_dx: int, max_dy: int) -> Union[np.ndarray, np.ndarray]:
        """
            Random translation function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                mask       (np.ndarray) : (H,W,1) Image.
                obj_mask   (np.ndarray) : (H,W,1) Image.
                min_dx  (int)      : Minimum value of pixel movement distance based on the x-axis when translating an image.
                min_dy  (int)      : Minimum value of pixel movement distance based on the y-axis when translating an image.
                max_dx  (int)      : Maximum value of pixel movement distance based on the x-axis when translating an image.
                max_dy  (int)      : Maximum value of pixel movement distance based on the y-axis when translating an image.
                
        """
        random_dx = random.randint(min_dx, max_dx)
        random_dy = random.randint(min_dy, max_dy)

        rows, cols = rgb.shape[:2]
        trans_mat = np.float64([[1, 0, random_dx], [0, 1, random_dy]])


        trans_rgb = cv2.warpAffine(rgb, trans_mat, (cols, rows))
        trans_mask = cv2.warpAffine(mask, trans_mat, (cols, rows))
        trans_obj_mask = cv2.warpAffine(obj_mask, trans_mat, (cols, rows))

        return trans_rgb, trans_mask, trans_obj_mask

    def image_random_rotation(self, rgb: np.ndarray, mask: np.ndarray, obj_mask: np.ndarray,
                              rot_min: int = 10, rot_max: int = 45) -> Union[np.ndarray, np.ndarray]:
        """
            Random rotation function   
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                rot_min  (int)        : Minimum rotation angle (degree).
                rot_max  (int)        : Maximum rotation angle (degree).
        """
        rot = random.randint(rot_min, rot_max)
        reverse = random.randint(1, 2)

        if reverse == 2:
            rot *= -1

        h, w = rgb.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
        rgb = cv2.warpAffine(rgb, rot_mat, (w, h))
        mask = cv2.warpAffine(mask, rot_mat, (w, h))
        obj_mask = cv2.warpAffine(obj_mask, rot_mat, (w, h))
        
        return rgb, mask, obj_mask


    def get_coords_from_mask(self, rgb: np.ndarray, mask: np.ndarray, obj_mask: np.ndarray):
        """
            Returns the coordinates (x_min, y_min_, x_max, y_max) of an object in a mask image.
            Args:
                rgb       (np.ndarray) : (H,W,3) Image.
                mask      (np.ndarray) : (H,W,1) Image.
                obj_mask  (np.ndarray) : (H,W,1) Image.
        """
        sample_labels = []
        sample_bboxes = []

        shape_h, shape_w = rgb.shape[:2]

        for label_batch in self.label_list:  # 1 ~ obj nums
            rgb_idx = label_batch['rgb'][2]
            
            binary_mask = np.where(mask == rgb_idx, 255, 0)

            binary_mask = binary_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calculate the bounding box of the area to be synthesized using the contour of the mask.
                x, y, w, h = cv2.boundingRect(contour)
                
                # convert y_min, x_min, y_max, x_max
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                # Scaling 0 ~ 1
                x_min /= shape_w
                y_min /= shape_h
                x_max /= shape_w
                y_max /= shape_h
                
                sample_bboxes.append([y_min, x_min, y_max, x_max])

                for idx in range(len(self.label_list)):
                    pixel_rgb = self.label_list[idx]['rgb']
                    pixel_value = pixel_rgb[2] # BGR

                    if np.max(mask[y:y+h, x:x+w]) == pixel_value:
                        # 'idx-1' is ignore background class
                        # only use foreground class (0: foreground 1: foreground_2 ..)
                        sample_labels.append(idx-1)
        
        # if len(sample_bboxes) >= 5:
            
            
        #     print(sample_bboxes)

            
        #     plt.imshow(mask)
        #     plt.show()

        return rgb, sample_bboxes, sample_labels


    def save_samples(self, rgb: np.ndarray, bbox: list, labels: list, prefix: str):
        """
            Save image and mask
            Args:
                rgb     (np.ndarray) : (H,W,3) Image.
                bbox    (list)       : List type [y_min, x_min, y_max, x_max].
                labels  (list)       : List type Integer labels.
                prefix  (str)        : The name of the image to be saved.
        """

        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_.png', rgb)

        with open(self.OUT_BBOX_PATH + prefix +'_.txt', "w") as file:
            for i in range(len(bbox)):
                file.writelines(str(bbox[i]) + '\n')

        with open(self.OUT_LABEL_PATH + prefix +'_.txt', "w") as file:
            for i in range(len(labels)):
                file.writelines(str(labels[i]) + '\n')


        # with open(self.OUT_BBOX_PATH + prefix +'_.txt', "r") as file:
        #     bbox_list = file.readlines()
            
        # with open(self.OUT_LABEL_PATH + prefix +'_.txt', "r") as file:
        #     label_list = file.readlines()
        
        
        # for i in range(len(label_list)):
        #     bbox_list[i] = bbox_list[i].replace('\n', '')
        #     label_list[i] = label_list[i].replace('\n', '')

        #     label_list[i] = int(label_list[i])

        #     # bbox_list[i] # str, x1, y1, x2, y2
        #     bbox_list[i] = bbox_list[i].replace('[', '')
        #     bbox_list[i] = bbox_list[i].replace(']', '')
        #     bbox_batch = bbox_list[i].split(',')

        #     batch_box_out = []
        #     for j in range(len(bbox_batch)):
        #         bbox_batch[j] = bbox_batch[j].replace(' ', '')    
        #         batch_box_out.append(float(bbox_batch[j]))
            
        #     bbox_list[i] = batch_box_out
            
        # bbox_list = np.array(bbox_list)
        # label_list = np.array(label_list)
        

    def plot_images(self, rgb: np.ndarray, mask: np.ndarray):
        """
            Image and mask plotting on screen function.
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                aspect_ratio  (float)        : Image Aspect Ratio. Code is written vertically.
        """
        rows = 1
        cols = 3

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # convert to unit8 type
        rgb = rgb.astype(np.uint8)
        mask = mask.astype(np.uint8)

        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(rgb)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(mask)
        ax0.set_title('mask')
        ax0.axis("off")

        mask = tf.concat([mask, mask, mask], axis=-1)

        draw_mask = tf.where(mask >= 1, mask, rgb)

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(draw_mask)
        ax0.set_title('draw_mask')
        ax0.axis("off")
        plt.show()
        plt.close()


    def draw_bounding(self, img, bboxes, labels, img_size, label_list):
        """
        bbox = np.array(bbox)
        label = np.array(label)
        convert_boxes = bbox.copy()
        convert_boxes[:, [0,2]] = bbox.copy()[:, [1,3]]
        convert_boxes[:, [1,3]] = bbox.copy()[:, [0,2]]
        image_loader.draw_bounding(img= original_rgb, bboxes=convert_boxes, labels=label, img_size=original_rgb.shape[:2], label_list=TEST_CLASSES)
        """
        # resizing 작업
        if np.max(bboxes) < 10:
            bboxes[:, [0,2]] = bboxes[:, [0,2]]*img_size[1]
            bboxes[:, [1,3]] = bboxes[:, [1,3]]*img_size[0]
        

        for i, bbox in enumerate(bboxes):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            img_box = np.copy(img)
            color = (127, 127, 127) * labels[i]
            cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_box, label_list[int(labels[i])], (xmin + 5, ymin - 5), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            alpha = 0.8
            cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)
                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    TEST_CLASSES = ['0', '1']
    

    image_loader = MaskDataGenerator(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    obj_mask_list = image_loader.get_obj_mask_list()
    
        

    for idx in range(len(rgb_list)):
        
        original_rgb = cv2.imread(rgb_list[idx])
        original_mask = cv2.imread(mask_list[idx])
        original_obj_mask = cv2.imread(obj_mask_list[idx])

        if np.max(original_mask) == 0:
            print('no labels')
            continue
        print(idx)
        original_mask = original_mask[:, :, :1]
        
        rgb, mask, obj_mask = image_loader.image_resize(
            rgb=original_rgb, mask=original_mask, obj_mask=original_obj_mask, size=(1280, 720))

        original_rgb, bbox, label = image_loader.get_coords_from_mask(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=original_rgb, bbox=bbox, labels=label, prefix='original_{0}'.format(idx))

        equal_rgb = image_loader.image_histogram_equalization(rgb=rgb.copy())
        equal_rgb, equal_bbox, equal_label = image_loader.get_coords_from_mask(rgb=equal_rgb, mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=equal_rgb, bbox=equal_bbox, labels=equal_label, prefix='histogram_equal_{0}'.format(idx))

        blur_rgb = image_loader.image_random_bluring(rgb=rgb.copy())
        blur_rgb, blur_bbox, blur_label = image_loader.get_coords_from_mask(rgb=blur_rgb, mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=blur_rgb, bbox=blur_bbox, labels=blur_label, prefix='blur_{0}'.format(idx))

        rot_rgb, rot_mask, rot_obj_mask = image_loader.image_random_rotation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        rot_rgb, rot_bbox, rot_label = image_loader.get_coords_from_mask(rgb=rot_rgb, mask=rot_mask, obj_mask=rot_obj_mask)
        image_loader.save_samples(rgb=rot_rgb, bbox=rot_bbox, labels=rot_label, prefix='rot_{0}'.format(idx))

        trans_rgb, trans_mask, trans_obj_mask = image_loader.image_random_translation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy(), min_dx=10, min_dy=20, max_dx=100, max_dy=200)
        trans_rgb, trans_bbox, trans_label = image_loader.get_coords_from_mask(rgb=trans_rgb, mask=trans_mask, obj_mask=trans_obj_mask)
        image_loader.save_samples(rgb=trans_rgb, bbox=trans_bbox, labels=trans_label, prefix='trans_{0}'.format(idx))
        