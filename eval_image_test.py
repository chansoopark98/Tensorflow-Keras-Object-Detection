import tensorflow as tf
import cv2
import numpy as np
# im id 379842

img = cv2.imread('test/000000379842.jpg')
img_shape = img.shape[:2]
print(img_shape)
# GT
gt_array = []
gt_array.append([369.71,2.43,194.16,316.31])
gt_array.append([343.66,20.72,276.03,339.28])
gt_array.append([7.5,1.41,307.5,352.5])
gt_array.append([270.7,22.32,66.74,329.93])


def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

def wh2minmax(array):
    x, y, w, h = array
    xmin = int(x)
    ymin = int(y)
    xmax = int(w + xmin)
    ymax = int(h + ymin)

    return xmin, ymin, xmax, ymax

    # x = round(float(xmin), 2)
    # y = round(float(ymin), 2)
    # w = round(float((xmax - xmin) + 1), 2)
    # h = round(float((ymax - ymin) + 1), 2)
    #
    # xmin = xmin * img_shapes[index][1]
    # ymin = ymin * img_shapes[index][0]
    # xmax = xmax * img_shapes[index][1]
    # ymax = ymax * img_shapes[index][0]


img_box = np.copy(img)


color = (52, 151, 51)

for i in range(len(gt_array)):
    array = gt_array[i]



    xmin, ymin, xmax, ymax = wh2minmax(array)
    print(xmin, ymin, xmax, ymax)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img_box, COCO_CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
    #             (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)
cv2.imwrite('test/output.jpg', img)

