import tensorflow as tf
import numpy as np
import itertools
from typing import Any, Optional
import tensorflow_addons as tfa
import math
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy, Huber

_EPSILON = tf.keras.backend.epsilon()

@tf.keras.utils.register_keras_serializable()
class DetectionLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, global_batch_size: int,
                 use_multi_gpu: bool = False, use_focal: bool = True,
                 **kwargs):
        """
        Args:
            Define the classification loss and the bounding box regression loss.
              
            num_classes       (int)  : Number of classes to classify 
                                       (must be equal to number of last filters in the model).
            global_batch_size (int)  : Global batch size (Batch_size = GLOBAL_BATCH_SIZE / GPUs).
            use_multi_gpu     (bool) : To calculate the loss for each gpu when using distributed training.
            use_focal     (bool) : Use Focal Cross entropy loss.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.global_batch_size = global_batch_size
        self.use_multi_gpu = use_multi_gpu
        self.use_focal = use_focal

    def get_config(self):
        """
            Returns the config dictionary for a Loss instance.
        """
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits, use_multi_gpu=self.use_multi_gpu,
                      global_batch_size=self.global_batch_size, num_classes=self.num_classes)
        return config

    def ce_with_hard_negaitve_mining(self, confidence: tf.Tensor, labels: tf.Tensor):
        loss = -tf.nn.log_softmax(confidence, axis=2)[:, :, 0]
        loss = tf.stop_gradient(loss)

        mask = self.hard_negative_mining(loss, labels, neg_pos_ratio=3.0)
        mask = tf.stop_gradient(mask) # neg sample 마스크

        confidence = tf.boolean_mask(confidence, mask)

        logits = tf.reshape(confidence, [-1, self.num_classes])
        labels = tf.boolean_mask(labels, mask)

        # calc classification loss
        # logits (samples, classes) labels (samples)
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)
        classification_loss = tf.math.reduce_sum(ce_loss)

        return classification_loss

    def giou_loss(self, target, output):
        giou_loss = tfa.losses.giou_loss(target, output)

        return giou_loss

    
    def ciou_loss(self, boxes1_x0y0x1y1, boxes2_x0y0x1y1):
        eps = tf.keras.backend.epsilon()

        # x,y,w,h -> x0y0x1y1
        xywh_format = boxes1_x0y0x1y1.numpy()

        boxes1_x0y0x1y1 = tf.maximum(boxes1_x0y0x1y1, eps)
        boxes1_x0y0x1y1 = tf.stack([boxes1_x0y0x1y1[..., 0],
                                    boxes1_x0y0x1y1[..., 1],
                                    boxes1_x0y0x1y1[..., 0] + boxes1_x0y0x1y1[..., 2],
                                    boxes1_x0y0x1y1[..., 1] + boxes1_x0y0x1y1[..., 3]], axis=1)
        x1y1x2y2 = boxes1_x0y0x1y1.numpy()

        boxes2_x0y0x1y1 = tf.maximum(boxes2_x0y0x1y1, eps)
        boxes2_x0y0x1y1 = tf.stack([boxes2_x0y0x1y1[..., 0],
                                    boxes2_x0y0x1y1[..., 1],
                                    boxes2_x0y0x1y1[..., 0] + boxes2_x0y0x1y1[..., 2],
                                    boxes2_x0y0x1y1[..., 1] + boxes2_x0y0x1y1[..., 3]], axis=1)
        
        np_boxes2 = boxes2_x0y0x1y1.numpy()

        # area
        boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                    boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
        np_boxes1_area = boxes1_area.numpy()
        

        boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                    boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])
        
        np_boxes2_area = boxes2_area.numpy()

        # top-left and bottom-right coord, shape: (8, 13, 13, 3, 2)
        left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])
        

        # intersection area and iou
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        np_inter_area = inter_area.numpy()

        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-9)
        np_iou = iou.numpy()
        iou = tf.abs(iou)
        np_iou = iou.numpy()

        # top-left and bottom-right coord of the enclosing rectangle, shape: (8, 13, 13, 3, 2)
        enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

        # diagnal ** 2
        enclose_wh = enclose_right_down - enclose_left_up
        enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)
        np_enclose_c2 = enclose_c2.numpy()

        # center distances between two rectangles
        x_distance = K.pow(boxes1_x0y0x1y1[..., 0] - boxes2_x0y0x1y1[..., 0], 2)
        np_x_distance = x_distance.numpy()
        y_distance = K.pow(boxes1_x0y0x1y1[..., 1] - boxes2_x0y0x1y1[..., 1], 2)
        y_distance = y_distance.numpy()
        p2 = (x_distance + y_distance) + tf.keras.backend.epsilon()
        np_p2 = p2.numpy()

        # add av
        
        # atan1 = tf.atan((boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) / (boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1]) + tf.keras.backend.epsilon()) # w, h
        atan1 = tf.math.divide_no_nan((boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]), (boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])) + tf.keras.backend.epsilon() # w, h
        np_atan1 = atan1.numpy()
        # atan2 = tf.atan((boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) / (boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1]) + tf.keras.backend.epsilon())
        atan2 = tf.math.divide_no_nan((boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]), (boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])) + tf.keras.backend.epsilon() # w, h
        np_atan2 = atan2.numpy()
        v = 4.0 * (K.pow(atan1 - atan2, 2) + tf.keras.backend.epsilon()) / (math.pi ** 2)
        np_v = v.numpy()
        a = v / (1 - iou + v)
        np_a = a.numpy()

        ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
        np_ciou = ciou.numpy()
        
        sub_distance = p2 / enclose_c2
        sub_distance = sub_distance.numpy()

        sub_a_v = (1.0 * a * v)
        sub_a_v = sub_a_v.numpy()

        ciou = iou - sub_distance - sub_a_v
        np_ciou = ciou.numpy()
        return 1 - ciou
        
        

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_pred (loc, conf, obj)
        # y_true (labels, loc, obj)

        # y_true
        labels = tf.math.argmax(y_true[:,:,:self.num_classes], axis=2)
        # gt_locations = y_true[:,:,self.num_classes:]
        gt_locations = y_true[:,:,self.num_classes:-1] # Test objectness
        true_obj = y_true[:, :, -1:]

        
        # y_pred
        
        confidence = y_pred[:,:,:self.num_classes]
        predicted_locations = y_pred[:,:,self.num_classes:-1]
        pred_obj = y_pred[:, :, -1:]

        if self.use_focal:
            # focal_labels : [batch * samples]
            focal_labels = tf.reshape(labels, [-1])
            # confidence : [batch * samples , classes]
            confidence = tf.reshape(confidence, [-1, self.num_classes])
            classification_loss = self.sparse_categorical_focal_loss(y_true=focal_labels, y_pred=confidence, gamma=2.0, from_logits=True)
            classification_loss = tf.reduce_sum(classification_loss)
        else:
            classification_loss = self.ce_with_hard_negaitve_mining(confidence=confidence, labels=labels)
        
        pos_mask = labels > 0
        predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
        gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])

        gt_locations = tf.where(tf.shape(gt_locations)[0] == 0, tf.zeros([1, 4]), gt_locations)

        # Test DIoU loss
        # giou_loss = self.giou_loss(target=gt_locations, output=predicted_locations)

        # calc localization loss
        # smooth_l1_loss = tf.math.reduce_sum(self.smooth_l1(scores=predicted_locations,labels=gt_locations))
        smooth_l1_loss = Huber()(y_true=gt_locations, y_pred=predicted_locations)

        ciou_loss = self.ciou_loss(gt_locations, predicted_locations)
        
        # num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
        num_pos = tf.cast(tf.where(tf.shape(gt_locations)[0] == 0, 1, tf.shape(gt_locations)[0]), tf.float32)

        # objectness loss
        obj_loss = BinaryCrossentropy(from_logits=False)(y_true=true_obj, y_pred=pred_obj)
        obj_loss *= 0.4

        # divide num_pos objects
        loc_loss = smooth_l1_loss / num_pos
        ciou_loss = ciou_loss / num_pos
        # giou_loss = giou_loss / num_pos
        class_loss = classification_loss / num_pos
        
        # Add to total loss
        mbox_loss = loc_loss + class_loss + obj_loss + ciou_loss

        # If use multi gpu, divide loss value by gpu numbers
        # if self.use_multi_gpu:
        #     mbox_loss *= (1. / self.global_batch_size)
            
        return mbox_loss


    def smooth_l1(self, labels: tf.Tensor, scores: tf.Tensor, sigma=1.0):
        """
        Generate a smooth L1 loss (Boundig box regression loss)

        labels    (tf.Tensor)  : A boundig box coordinates from groundtruth.
        scores    (tf.Tensor)  : A predict boundig box coordinates.
        """
        diff = scores-labels
        abs_diff = tf.abs(diff)
        
        clipping = tf.less(abs_diff, 1/(sigma**2))
        choose_value = tf.where(clipping, 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))
        return choose_value


    def hard_negative_mining(self, loss: tf.Tensor, labels: tf.Tensor, neg_pos_ratio: float):
        pos_mask = labels > 0
        num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True)
        num_neg = num_pos * neg_pos_ratio

        loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss)

        indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
        orders = tf.argsort(indexes, axis=1)
        neg_mask = tf.cast(orders, tf.float32) < num_neg

        return tf.logical_or(pos_mask ,neg_mask)


    def sparse_categorical_focal_loss(self, y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1,
                                  ) -> tf.Tensor:
        # Process focusing parameter
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        gamma_rank = gamma.shape.rank
        scalar_gamma = gamma_rank == 0

        # Process class weight
        if class_weight is not None:
            class_weight = tf.convert_to_tensor(class_weight,
                                                dtype=tf.dtypes.float32)

        # Process prediction tensor
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred_rank = y_pred.shape.rank
        if y_pred_rank is not None:
            axis %= y_pred_rank
            if axis != y_pred_rank - 1:
                # Put channel axis last for sparse_softmax_cross_entropy_with_logits
                perm = list(itertools.chain(range(axis),
                                            range(axis + 1, y_pred_rank), [axis]))
                y_pred = tf.transpose(y_pred, perm=perm)
        elif axis != -1:
            raise ValueError(
                f'Cannot compute sparse categorical focal loss with axis={axis} on '
                'a prediction tensor with statically unknown rank.')
        y_pred_shape = tf.shape(y_pred)

        # Process ground truth tensor
        y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
        y_true_rank = y_true.shape.rank

        if y_true_rank is None:
            raise NotImplementedError('Sparse categorical focal loss not supported '
                                    'for target/label tensors of unknown rank')

        reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                        y_pred_rank != y_true_rank + 1)
        if reshape_needed:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

        if from_logits:
            logits = y_pred
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred
            logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits)

        y_true_rank = y_true.shape.rank
        probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

        if not scalar_gamma:
            gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
        focal_modulation = (1 - probs) ** gamma

        loss = focal_modulation * xent_loss

        if class_weight is not None:
            class_weight = tf.gather(class_weight, y_true, axis=0,
                                    batch_dims=y_true_rank)
            loss *= class_weight

        if reshape_needed:
            loss = tf.reshape(loss, y_pred_shape[:-1])

        return loss