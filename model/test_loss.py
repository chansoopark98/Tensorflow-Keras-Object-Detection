import tensorflow as tf
import numpy as np
import itertools
from typing import Any, Optional

_EPSILON = tf.keras.backend.epsilon()

@tf.keras.utils.register_keras_serializable()
class DetectionLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, global_batch_size: int,
                 use_multi_gpu: bool = False, use_focal: bool = False,
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


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        labels = tf.math.argmax(y_true[:,:,:self.num_classes], axis=2)
        confidence = y_pred[:,:,:self.num_classes]
        predicted_locations = y_pred[:,:,self.num_classes:]
        gt_locations = y_true[:,:,self.num_classes:]
        
        ce_logits = confidence
    
        ce_loss = self.sparse_categorical_focal_loss(y_true=labels, y_pred=ce_logits, gamma=2.0, from_logits=True)

        # ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ce_logits,
        #                                                              labels=labels)
        classification_loss = tf.math.reduce_sum(ce_loss)

        pos_mask = labels > 0
        predicted_locations = tf.reshape(tf.boolean_mask(predicted_locations, pos_mask), [-1, 4])
        gt_locations = tf.reshape(tf.boolean_mask(gt_locations, pos_mask), [-1, 4])

        # calc localization loss
        smooth_l1_loss = tf.math.reduce_sum(self.smooth_l1(scores=predicted_locations,labels=gt_locations))
        num_pos = tf.cast(tf.shape(gt_locations)[0], tf.float32)
        
        # divide num_pos objects
        loc_loss = smooth_l1_loss / num_pos
        class_loss = classification_loss / num_pos

        if self.use_multi_gpu:
            loc_loss *= (1. / self.global_batch_size)
            class_loss *= (1. / self.global_batch_size)
            
        mbox_loss = loc_loss + class_loss
        return mbox_loss


    def smooth_l1(self, labels: tf.Tensor, scores: tf.Tensor, sigma=1.0):
        """
        Generate a smooth L1 loss (Boundig box regression loss)

        labels    (tf.Tensor)  : A boundig box coordinates from groundtruth.
        scores    (tf.Tensor)  : A predict boundig box coordinates.
        """
        diff = scores-labels
        abs_diff = tf.abs(diff)

        return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))


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