from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import random
import math

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

        self.TRAIN_RGB_PATH = self.OUTPUT_PATH + 'train/rgb/'
        self.TRAIN_LABEL_PATH = self.OUTPUT_PATH + 'train/label/'
        self.TRAIN_BBOX_PATH = self.OUTPUT_PATH + 'train/bbox/'
        os.makedirs(self.TRAIN_RGB_PATH, exist_ok=True)
        os.makedirs(self.TRAIN_LABEL_PATH, exist_ok=True)
        os.makedirs(self.TRAIN_BBOX_PATH, exist_ok=True)

        self.VALID_RGB_PATH = self.OUTPUT_PATH + 'validation/rgb/'
        self.VALID_LABEL_PATH = self.OUTPUT_PATH + 'validation/label/'
        self.VALID_BBOX_PATH = self.OUTPUT_PATH + 'validation/bbox/'
        os.makedirs(self.VALID_RGB_PATH, exist_ok=True)
        os.makedirs(self.VALID_LABEL_PATH, exist_ok=True)
        os.makedirs(self.VALID_BBOX_PATH, exist_ok=True)


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
        radian = rot * math.pi / 180.0 

        rgb = tfa.image.rotate(images=rgb, angles=radian).numpy()
        mask = tfa.image.rotate(images=mask, angles=radian).numpy()
        obj_mask = tfa.image.rotate(images=obj_mask, angles=radian).numpy()

        return rgb, mask, obj_mask


    def image_random_crop(self, rgb: np.ndarray, mask: np.ndarray, obj_mask: np.ndarray,
                          ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
            Random crop function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                mask       (np.ndarray) : (H,W,1) Image.
                obj_mask   (np.ndarray) : (H,W,1) Image.
        """
        original_h, original_w = rgb.shape[:2]
        aspect_ratio = original_h / original_w

        widht_scale = tf.random.uniform([], 0.7, 0.95)
        
        new_w = original_w * widht_scale
        new_h = new_w * aspect_ratio
        
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, axis=-1)
        if len(obj_mask.shape) == 2:
            obj_mask = tf.expand_dims(obj_mask, axis=-1)
        
        concat_img = tf.concat([rgb, mask, obj_mask], axis=-1)
        concat_img = tf.image.random_crop(concat_img, size=[new_h, new_w, 7])
        
        crop_img = concat_img[:, :, :3].numpy()
        crop_mask = concat_img[:, :, 3].numpy()
        crop_obj_mask = concat_img[:, :, 4:].numpy()

        return crop_img, crop_mask, crop_obj_mask


    def image_random_padding(self, rgb: np.ndarray, mask: np.ndarray, obj_mask: np.ndarray
                            ) -> Union[np.ndarray, np.ndarray, np.ndarray]:

        top_pad = random.randint(0, 100)
        bottom_pad = random.randint(0, 100)
        left_pad = random.randint(0, 100)
        right_pad = random.randint(0, 100)

        rgb = cv2.copyMakeBorder(rgb, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        mask = cv2.copyMakeBorder(mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0])
        obj_mask = cv2.copyMakeBorder(obj_mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0])

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
        
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
        obj_mask = obj_mask.astype(np.uint8)

        obj_idx = np.unique(obj_mask)
        obj_idx = np.delete(obj_idx, 0)
        

        for target_value in obj_idx:  # 1 ~ obj nums
            binary_mask = np.where(obj_mask==target_value, 255, 0)

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
                        sample_labels.append(idx)
        
        return rgb, sample_bboxes, sample_labels


    def save_samples(self, rgb: np.ndarray, bbox: list, labels: list, prefix: str, save_mode: str = 'train'):
        """
            Save image and mask
            Args:
                rgb       (np.ndarray) : (H,W,3) Image.
                bbox      (list)       : List type [y_min, x_min, y_max, x_max].
                labels    (list)       : List type Integer labels.
                prefix    (str)        : The name of the image to be saved.
                save_mode (str)        : Save mode (train or validation)
        """
        bbox_len = len(bbox)
        labels_len = len(labels)
        if bbox_len == 0:
            print('bbox len is 0', bbox_len)
        elif labels_len == 0:
            print('labels len is 0', labels_len)
        elif bbox_len != labels_len:
            print('bbox and label size does not match')
        else:
            if save_mode == 'train':
                save_rgb_path = self.TRAIN_RGB_PATH
                save_label_path = self.TRAIN_LABEL_PATH
                save_bbox_path = self.TRAIN_BBOX_PATH
            else:
                save_rgb_path = self.VALID_RGB_PATH
                save_label_path = self.VALID_LABEL_PATH
                save_bbox_path = self.VALID_BBOX_PATH

            cv2.imwrite(save_rgb_path + prefix +'_.png', rgb)

            with open(save_bbox_path + prefix +'_.txt', "w") as file:
                for i in range(len(bbox)):
                    file.writelines(str(bbox[i]) + '\n')

            with open(save_label_path + prefix +'_.txt', "w") as file:
                for i in range(len(labels)):
                    file.writelines(str(labels[i]) + '\n')

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
    
    # 같은 index로 랜덤 셔플링
    data_len = len(rgb_list)
    print(data_len)
    key = np.arange(data_len)
    np.random.shuffle(key)
    print(key)
    rgb_list = np.array(rgb_list)[key]
    rgb_list = list(rgb_list)
    
    mask_list = np.array(mask_list)[key]
    mask_list = list(mask_list)

    obj_mask_list = np.array(obj_mask_list)[key]
    obj_mask_list = list(obj_mask_list)


    validation_split = int(data_len * 0.1)
    
    train_rgb_list = rgb_list[validation_split:]
    valid_rgb_list = rgb_list[:validation_split]

    train_mask_list = mask_list[validation_split:]
    valid_mask_list = mask_list[:validation_split]

    train_obj_mask_list = obj_mask_list[validation_split:]
    valid_obj_mask_list = obj_mask_list[:validation_split]


    for idx in range(len(train_rgb_list)):
        
        original_rgb = cv2.imread(train_rgb_list[idx])
        original_mask = cv2.imread(train_mask_list[idx])
        original_obj_mask = cv2.imread(train_obj_mask_list[idx])

        if np.max(original_mask) == 0:
            print('no labels')
            continue

        print(idx)
        original_mask = original_mask[:, :, :1]
        
        original_h, original_w = original_rgb.shape[:2]
        aspect_ratio = original_h / original_w
        original_w *= 0.35
        original_h = original_w * aspect_ratio
        
        # rgb = original_rgb.copy()
        # mask = original_mask.copy()
        # obj_mask = original_obj_mask.copy()



        rgb, mask, obj_mask = image_loader.image_resize(
            rgb=original_rgb, mask=original_mask, obj_mask=original_obj_mask, size=(640, 360))
        original_rgb, bbox, label = image_loader.get_coords_from_mask(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=original_rgb, bbox=bbox, labels=label, prefix='original_{0}'.format(idx))

        for pad_idx in range(2):
            pad_rgb, pad_mask, pad_obj_mask = image_loader.image_random_padding(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
            pad_rgb, pad_box, pad_label = image_loader.get_coords_from_mask(rgb=pad_rgb, mask=pad_mask, obj_mask=pad_obj_mask)
            image_loader.save_samples(rgb=pad_rgb, bbox=pad_box, labels=pad_label, prefix='pad_{0}_{1}'.format(idx, pad_idx))

        equal_rgb = image_loader.image_histogram_equalization(rgb=rgb.copy())
        equal_rgb, equal_bbox, equal_label = image_loader.get_coords_from_mask(rgb=equal_rgb, mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=equal_rgb, bbox=equal_bbox, labels=equal_label, prefix='histogram_equal_{0}'.format(idx))

        blur_rgb = image_loader.image_random_bluring(rgb=rgb.copy(), gaussian_min=3, gaussian_max=21)
        blur_rgb, blur_bbox, blur_label = image_loader.get_coords_from_mask(rgb=blur_rgb, mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=blur_rgb, bbox=blur_bbox, labels=blur_label, prefix='blur_{0}'.format(idx))

        for rotate_idx in range(2):
            rot_rgb, rot_mask, rot_obj_mask = image_loader.image_random_rotation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
            rot_rgb, rot_bbox, rot_label = image_loader.get_coords_from_mask(rgb=rot_rgb, mask=rot_mask, obj_mask=rot_obj_mask)
            image_loader.save_samples(rgb=rot_rgb, bbox=rot_bbox, labels=rot_label, prefix='rot_{0}_idx_{1}'.format(idx, rotate_idx))

        trans_rgb, trans_mask, trans_obj_mask = image_loader.image_random_translation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy(), min_dx=10, min_dy=20, max_dx=50, max_dy=100)
        trans_rgb, trans_bbox, trans_label = image_loader.get_coords_from_mask(rgb=trans_rgb, mask=trans_mask, obj_mask=trans_obj_mask)
        image_loader.save_samples(rgb=trans_rgb, bbox=trans_bbox, labels=trans_label, prefix='trans_{0}'.format(idx))


        for crop_idx in range(3):
            crop_rgb, crop_mask, crop_obj_mask = image_loader.image_random_crop(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
            crop_rgb, crop_bbox, crop_label = image_loader.get_coords_from_mask(rgb=crop_rgb, mask=crop_mask, obj_mask=crop_obj_mask)
            image_loader.save_samples(rgb=crop_rgb, bbox=crop_bbox, labels=crop_label, prefix='crop_{0}_idx_{1}'.format(idx, crop_idx))
    

    for idx in range(len(valid_rgb_list)):
        
        original_rgb = cv2.imread(valid_rgb_list[idx])
        original_mask = cv2.imread(valid_mask_list[idx])
        original_obj_mask = cv2.imread(valid_obj_mask_list[idx])

        if np.max(original_mask) == 0:
            print('no labels')
            continue

        print(idx)
        original_mask = original_mask[:, :, :1]
        
        original_h, original_w = original_rgb.shape[:2]
        aspect_ratio = original_h / original_w
        original_w *= 0.35
        original_h = original_w * aspect_ratio        
        
        rgb, mask, obj_mask = image_loader.image_resize(
            rgb=original_rgb, mask=original_mask, obj_mask=original_obj_mask, size=(640, 360))

        pad_rgb, pad_mask, pad_obj_mask = image_loader.image_random_padding(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        pad_rgb, pad_box, pad_label = image_loader.get_coords_from_mask(rgb=pad_rgb, mask=pad_mask, obj_mask=pad_obj_mask)
        image_loader.save_samples(rgb=pad_rgb, bbox=pad_box, labels=pad_label, prefix='valid_pad_{0}_'.format(idx), save_mode='valid')
        

        original_rgb, bbox, label = image_loader.get_coords_from_mask(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        image_loader.save_samples(rgb=original_rgb, bbox=bbox, labels=label, prefix='original_{0}'.format(idx), save_mode='valid')


        rot_rgb, rot_mask, rot_obj_mask = image_loader.image_random_rotation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        rot_rgb, rot_bbox, rot_label = image_loader.get_coords_from_mask(rgb=rot_rgb, mask=rot_mask, obj_mask=rot_obj_mask)
        image_loader.save_samples(rgb=rot_rgb, bbox=rot_bbox, labels=rot_label, prefix='valid_rot_{0}_'.format(idx), save_mode='valid')

        trans_rgb, trans_mask, trans_obj_mask = image_loader.image_random_translation(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy(), min_dx=10, min_dy=20, max_dx=50, max_dy=100)
        trans_rgb, trans_bbox, trans_label = image_loader.get_coords_from_mask(rgb=trans_rgb, mask=trans_mask, obj_mask=trans_obj_mask)
        image_loader.save_samples(rgb=trans_rgb, bbox=trans_bbox, labels=trans_label, prefix='valid_trans_{0}'.format(idx), save_mode='valid')


    
        crop_rgb, crop_mask, crop_obj_mask = image_loader.image_random_crop(rgb=rgb.copy(), mask=mask.copy(), obj_mask=obj_mask.copy())
        crop_rgb, crop_bbox, crop_label = image_loader.get_coords_from_mask(rgb=crop_rgb, mask=crop_mask, obj_mask=crop_obj_mask)
        image_loader.save_samples(rgb=crop_rgb, bbox=crop_bbox, labels=crop_label, prefix='valid_crop_{0}'.format(idx), save_mode='valid')