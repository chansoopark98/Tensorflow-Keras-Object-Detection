import sys
from tensorflow import keras
import tensorflow as tf
import numpy as np
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()


def sparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1
                                  ) -> tf.Tensor:
    # Process focusing parameter
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)  # ()
    gamma_rank = gamma.shape.rank  # 0
    scalar_gamma = gamma_rank == 0  # True

    # Process class weight - 사용 x
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight,
                                            dtype=tf.dtypes.float32)

    # Process prediction tensor
    #y_pred = tf.convert_to_tensor(y_pred)  # B, 21
    y_pred_rank = y_pred.shape.rank  # RANK = 2
    if y_pred_rank is not None:
        axis %= y_pred_rank  # axis = 1

        if axis != y_pred_rank - 1:  # 실행 안함
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis),
                                        range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)

    elif axis != -1:  # 실행 안함
        raise ValueError(
            f'Cannot compute sparse categorical focal loss with axis={axis} on '
            'a prediction tensor with statically unknown rank.')
    y_pred_shape = tf.shape(y_pred)  # y_pred_shape ==> (2,)

    # Process ground truth tensor
    #y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank  # rank = 1

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported '
                                  'for target/label tensors of unknown rank')

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and  #
                      y_pred_rank != y_true_rank + 1)
    # 기존 코드
    # if reshape_needed:
    #     y_true = tf.reshape(y_true, [-1])
    #     y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])
    #
    # if from_logits: # this
    #     logits = y_pred
    #     probs = tf.nn.softmax(y_pred, axis=-1)
    # else:
    #     probs = y_pred
    #     logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:  # this
        logits = y_pred
        probs = -tf.nn.softmax(y_pred, axis=-1)

        # focal loss test 해볼거
        # focal_beta_loss에서 현재 reshape한거로 probs를 생성했는데
        # softmax 또는 log_softmax한 값으로 prbos를 추출해야함
        # 0513 기준으로 map 77까지는 나옴

    else:
        probs = y_pred # None, 21
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON)) # None, 21

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank # 1
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

    tf.print("  gather 이후 probs -1 ", probs, output_stream=sys.stdout)


    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
    focal_modulation = (1 - probs) ** gamma


    loss = focal_modulation * xent_loss
    loss = tf.reduce_sum(loss)

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0,
                                 batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        """
        Args:
            y_true : tensor-like, shape (N,)
                Integer class labels.
            y_pred : tensor-like, shape (N, K)
                Either probabilities or logits, depending on the `from_logits`
                parameter.
        Returns:
            :class:`tf.Tensor`
        """

        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             class_weight=self.class_weight,
                                             gamma=self.gamma,
                                             from_logits=self.from_logits)


def smooth_l1(labels, scores, sigma=1.0):
    diff = scores - labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1 / (sigma ** 2)), 0.5 * (sigma * diff) ** 2, abs_diff - 1 / (2 * sigma ** 2))


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """loss > softmax한 confidence"""
    pos_mask = labels > 0 # None, 16368
    num_pos = tf.math.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1, keepdims=True) # None, 1 예 > [[2][3]]
    num_neg = num_pos * neg_pos_ratio # [[2*3][3*3]]

    loss = tf.where(pos_mask, tf.convert_to_tensor(np.NINF), loss) # N, 16368 객체에 대해서만 INF


    indexes = tf.argsort(loss, axis=1, direction='DESCENDING') # N, N 내림차순 정렬

    orders = tf.argsort(indexes, axis=1) # N, N 오름차순 정렬

    neg_mask = tf.cast(orders, tf.float32) < num_neg # N, N

    return tf.logical_or(pos_mask, neg_mask)

import tensorflow_addons as tfa

from tensorflow.python.ops import array_ops


def focal_loss_on_object_detection(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    """
    focal loss = -alpha * (z-p)^gamma * log(p) - (1-alpha) * p^gamma * log(1-p)
    注（z-p)那一项，因为z是one-hot编码，公式其他部分都正常对应

    Args:
        prediction_tensor: [batch_size, num_anchors, num_classes]，one-hot表示
         target_tensor: [batch_size, num_anchors, num_classes] one-hot表示
        weights: [batch_size, num_anchors]
        alpha: focal loss超参数
        gamma: focal loss超参数
    Returns:
        loss: 返回loss的tensor常量
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # 对于positive prediction，只考虑前景部分的loss，背景loss为0
    # target_tensor > zeros <==> z=1, 所以positive系数 = z - p
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # 对于negative prediction，只考虑背景部分的loss，前景的为0
    # target_tensor > zeros <==> z=1, 所以negative系数 = 0
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)

def t_focal_loss(logits, labels, alpha=0.25, gamma=1.5):
    pos_pt = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-10, 0.999)
    fl = labels * tf.math.log(pos_pt) * tf.pow(1 - pos_pt, gamma) * alpha + (1 - labels) * tf.math.log(1 - pos_pt) * tf.pow(pos_pt, gamma) * (1 - alpha)
    fl = -tf.reduce_sum(fl, axis=1)
    return fl

def total_loss(y_true, y_pred, num_classes=21):
    labels = tf.argmax(y_true[:, :, :num_classes], axis=2)  # B, 16368
    predicted_locations = y_pred[:, :, num_classes:]  # B, None, 4
    gt_locations = y_true[:, :, num_classes:]  # B, 16368, None
    pos_mask = labels > 0  # B, 16368

    """
        y_true: (B, N, num_classes).
        y_pred:  (B, N, num_classes).     """


    epsilon = tf.keras.backend.epsilon()
    alpha = 0.25
    gamma = 1.5
    ce_true = y_true[:, :, :num_classes]
    ce_pred = y_pred[:, :, :num_classes]  # B, N, 21


    #cls_loss = focal_loss_on_object_detection(ce_pred, ce_true, alpha=0.25, gamma=1.5)
    cls_loss = t_focal_loss(ce_pred, ce_true)
    tf.print(" cls_loss => ", cls_loss, output_stream=sys.stdout, summarize=-1)


    """ tfa sigmoid focal loss"""
    # fl = tfa.losses.SigmoidFocalCrossEntropy()
    # ce_pred = tf.nn.sigmoid(ce_pred)
    # fl = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma, reduction=tf.keras.losses.Reduction.SUM)
    # test_focal_loss = fl(ce_true, ce_pred)
    # tf.print(" test_focal => ", test_focal_loss, output_stream=sys.stdout, summarize=-1)

    """ efficientdet focal loss"""
    # ce_pred = tf.clip_by_value(ce_pred, epsilon, 1.0 - epsilon)
    # alpha_factor = keras.backend.ones_like(ce_true) * alpha
    # alpha_factor = tf.where(keras.backend.equal(ce_true, 1), alpha_factor, 1 - alpha_factor)
    # focal_weight = tf.where(keras.backend.equal(ce_true, 1), 1 - ce_pred, ce_pred)
    # focal_weight = alpha_factor * focal_weight ** gamma
    # cls_loss = focal_weight * keras.backend.binary_crossentropy(ce_true, ce_pred)
    # cls_loss = tf.math.reduce_sum(cls_loss, axis=1)


    """ 0520 test focal"""
    #ce_pred = tf.clip_by_value(ce_pred, epsilon, 1.0 - epsilon)
    # cross_entropy = -ce_true * tf.math.log(ce_pred)
    # weight = alpha * ce_true * tf.math.pow((1 - ce_pred), gamma)
    # cls_loss = weight * cross_entropy
    # cls_loss = tf.math.reduce_sum(loss, axis=1)

    #tf.print(" focal  => ", loss, output_stream=sys.stdout)


    predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
    gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])

    # calc localization loss
    smooth_l1_loss = tf.math.reduce_sum(smooth_l1(scores=predicted_locations, labels=gt_locations))
    num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
    # divide num_pos objects
    loc_loss = smooth_l1_loss / num_pos
    focal_loss = cls_loss / num_pos
    mbox_loss = loc_loss + focal_loss
    return mbox_loss

