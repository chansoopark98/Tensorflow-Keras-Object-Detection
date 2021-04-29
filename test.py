import cv2
from config import *

specs = set_priorBox('b0')
print(specs)

img = cv2.imread('test/000000379842.jpg')

img_shape = img.shape[:2]
print(img_shape)

priors = create_priors_boxes(specs, 512)
w1 = priors[0][2].numpy() * img_shape[1]
h1 = priors[0][3].numpy() * img_shape[0]

w2 = priors[1][2].numpy() * img_shape[1]
h2 = priors[1][3].numpy() * img_shape[0]

w3 = priors[2][2].numpy() * img_shape[1]
h3 = priors[2][3].numpy() * img_shape[0]

w4 = priors[3][2].numpy() * img_shape[1]
h4 = priors[3][3].numpy() * img_shape[0]

w5 = priors[4][2].numpy() * img_shape[1]
h5 = priors[4][3].numpy() * img_shape[0]





# GT
gt_array = []
gt_array.append([img_shape[1]/2,img_shape[0]/2, w1, h1])
gt_array.append([img_shape[1]/2,img_shape[0]/2, w2, h2])
gt_array.append([img_shape[1]/2,img_shape[0]/2, w3, h3])
gt_array.append([img_shape[1]/2,img_shape[0]/2, w4, h4])
gt_array.append([img_shape[1]/2,img_shape[0]/2, w5, h5])




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
pred_color = (0, 0, 255)
for i in range(len(gt_array)):
    array = gt_array[i]



    xmin, ymin, xmax, ymax = wh2minmax(array)
    print(xmin, ymin, xmax, ymax)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    #cv2.rectangle(img_box, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img_box, COCO_CLASSES[int(labels[i]-1)], (xmin + 5, ymin - 5), font, 0.5,
    #             (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1. - alpha, 0, img)


cv2.imwrite('test/output.jpg', img)
