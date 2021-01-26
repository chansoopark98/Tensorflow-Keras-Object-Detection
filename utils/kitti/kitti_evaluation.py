#@title Evaluation
from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    경계 상자 간의 IoU (Intersection of Union)를 계산합니다.
     IoU는 교차 영역의 비율로 계산됩니다.
     그리고 노조의 영역.
     이 함수는 : obj :`numpy.ndarray` 및 : obj :`cupy.ndarray`를 모두 다음과 같이 허용합니다.
     입력. : obj :`bbox_a` 및 : obj :`bbox_b` 모두
     같은 유형.
     출력은 입력 유형과 동일한 유형입니다.
     Args :
         bbox_a (배열) : 형태가 : math :`(N, 4)`인 배열입니다.
             : math :`N`은 경계 상자의 수입니다.
             dtype은 : obj :`numpy.float32` 여야합니다.
         bbox_b (배열) : : obj :`bbox_a`와 유사한 배열,
             그 모양은 : math :`(K, 4)`입니다.
             dtype은 : obj :`numpy.float32` 여야합니다.
     보고:
         정렬:
         모양이 : math :`(N, K)`인 배열입니다. \
         인덱스 : math :`(n, k)`의 요소는 \ 사이의 IoU를 포함합니다.
         : obj :`bbox_a`의 : math :`n` 경계 상자 및 : math :`k` 경계 상자 \
         : obj :`bbox_b`의 상자.
     "" "
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def eval_detection(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults=None,
        iou_thresh=0.5,
        use_07_metric=False):
    """
    PASCAL VOC의 평가 코드를 기준으로 평균 정밀도를 계산합니다.
    이 함수는 데이터 세트에서 얻은 예측 경계 상자를 평가합니다.
    각 클래스의 평균 정밀도를 사용하여 : math :`N` 이미지가 있습니다.
    코드는 PASCAL VOC Challenge에서 사용되는 평가 코드를 기반으로합니다.
    Args :
        pred_bboxes (numpy.ndarray의 반복 가능) : : math :`N`의 반복 가능
            경계 상자 세트.
            색인은 기본 데이터 세트의 색인에 해당합니다.
            : obj :`pred_bboxes`의 각 요소는 좌표 집합입니다.
            경계 상자의. 모양이 : math :`(R, 4)`인 배열입니다.
            여기서 : math :`R`은
            상자에 따라 다를 수있는 경계 상자의 수입니다.
            두 번째 축은
            : math :`y_ {min}, x_ {min}, y_ {max}, x_ {max}`의 경계 상자입니다.
        pred_labels (numpy.ndarray의 반복 가능) : 레이블의 반복 가능.
            : obj :`pred_bboxes`와 유사하며 색인은
            기본 데이터 세트의 색인입니다. 길이는 : math :`N`입니다.
        pred_scores (numpy.ndarray의 반복 가능) : 신뢰의 반복 가능
            예측 된 경계 상자에 대한 점수. : obj :`pred_bboxes`와 유사합니다.
            색인은 기본 데이터 세트의 색인에 해당합니다.
            길이는 : math :`N`입니다.
        gt_bboxes (numpy.ndarray의 반복 가능) : 지상 진실의 반복 가능
            경계 상자
            길이는 : math :`N`입니다. : obj :`gt_bboxes`의 요소는
            모양이 : math :`(R, 4)`인 경계 상자. 수는
            각 이미지의 경계 상자는 숫자와 같을 필요가 없습니다.
            해당 예측 상자의.
        gt_labels (numpy.ndarray의 반복 가능) : Ground Truth의 반복 가능
            : obj :`gt_bboxes`와 유사하게 구성된 레이블.
        gt_difficults (numpy.ndarray 반복 가능) : 부울 반복 가능
            : obj :`gt_bboxes`와 유사하게 구성된 배열.
            이것은
            해당 Ground Truth 경계 상자는 어렵거나 그렇지 않습니다.
            기본적으로 이것은 : obj :`None`입니다. 이 경우이 기능은
            모든 경계 상자가 어렵지 않은 것으로 간주합니다.
        iou_thresh (float) : 교차점이 지나면 예측이 정확합니다.
            지상 진실과의 결합은이 값보다 높습니다.
        use_07_metric (bool) : PASCAL VOC 2007 평가 메트릭 사용 여부
            평균 정밀도를 계산합니다. 기본값은
            : obj :`거짓`.
    보고:
        dict :
        키, 값 유형 및 값 설명이 나열됩니다.
        이하.
        * ** ap ** (* numpy.ndarray *) : 평균 정밀도의 배열입니다. \
            : math :`l`-th 값은 평균 정밀도 \에 해당합니다.
            클래스 : math :`l`. class : math :`l`이 \에 존재하지 않는 경우
            : obj :`pred_labels` 또는 : obj :`gt_labels`, 해당 \
            값은 : obj :`numpy.nan`으로 설정됩니다.
        * ** map ** (* float *) : 클래스에 대한 평균 정밀도의 평균입니다.
    "" "
    """

    prec, rec = calc_kitti_pr_rc(pred_bboxes,
                                 pred_labels,
                                 pred_scores,
                                 gt_bboxes,
                                 gt_labels,
                                 gt_difficults,
                                 iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_kitti_pr_rc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """"" "PASCAL VOC의 평가 코드를 기준으로 정밀도 및 리콜을 계산합니다.
    이 함수는 정밀도와 재현율을 계산합니다.
    : math :`N`이있는 데이터 세트에서 얻은 예측 경계 상자
    이미지.
    코드는 PASCAL VOC Challenge에서 사용되는 평가 코드를 기반으로합니다.
    Args :
        pred_bboxes (numpy.ndarray의 반복 가능) : : math :`N`의 반복 가능
            경계 상자 세트.
            색인은 기본 데이터 세트의 색인에 해당합니다.
            : obj :`pred_bboxes`의 각 요소는 좌표 집합입니다.
            경계 상자의. 모양이 : math :`(R, 4)`인 배열입니다.
            여기서 : math :`R`은
            상자에 따라 다를 수있는 경계 상자의 수입니다.
            두 번째 축은
            : math :`y_ {min}, x_ {min}, y_ {max}, x_ {max}`의 경계 상자입니다.
        pred_labels (numpy.ndarray의 반복 가능) : 레이블의 반복 가능.
            : obj :`pred_bboxes`와 유사하며 색인은
            기본 데이터 세트의 색인입니다. 길이는 : math :`N`입니다.
        pred_scores (numpy.ndarray의 반복 가능) : 신뢰의 반복 가능
            예측 된 경계 상자에 대한 점수. : obj :`pred_bboxes`와 유사합니다.
            색인은 기본 데이터 세트의 색인에 해당합니다.
            길이는 : math :`N`입니다.
        gt_bboxes (numpy.ndarray의 반복 가능) : 지상 진실의 반복 가능
            경계 상자
            길이는 : math :`N`입니다. : obj :`gt_bboxes`의 요소는
            모양이 : math :`(R, 4)`인 경계 상자. 수는
            각 이미지의 경계 상자는 숫자와 같을 필요가 없습니다.
            해당 예측 상자의.
        gt_labels (numpy.ndarray의 반복 가능) : Ground Truth의 반복 가능
            : obj :`gt_bboxes`와 유사하게 구성된 레이블.
        gt_difficults (numpy.ndarray 반복 가능) : 부울 반복 가능
            : obj :`gt_bboxes`와 유사하게 구성된 배열.
            이것은
            해당 Ground Truth 경계 상자는 어렵거나 그렇지 않습니다.
            기본적으로 이것은 : obj :`None`입니다. 이 경우이 기능은
            모든 경계 상자가 어렵지 않은 것으로 간주합니다.
        iou_thresh (float) : 교차점이 지나면 예측이 정확합니다.
            지상 진실과의 결합은이 값 이상입니다 ..
    보고:
        두 목록의 튜플 :
        이 함수는 : obj :`prec` 및 : obj :`rec`의 두 목록을 반환합니다.
        * : obj :`prec` : 배열 목록입니다. : obj :`prec [l]`은 정밀도입니다.
            클래스 : math :`l`. class : math :`l`이 \에 존재하지 않는 경우
            : obj :`pred_labels` 또는 : obj :`gt_labels`, : obj :`prec [l]`은 \
            : obj :`None`으로 설정합니다.
        * : obj :`rec` : 배열 목록입니다. : obj :`rec [l]`은 회상입니다.
            클래스 : math :`l`. class : math :`l`이 \로 표시되지 않은 경우
            어려운 존재하지 않습니다 \
            : obj :`gt_labels`, : obj :`rec [l]`은 \
            : obj :`None`으로 설정합니다.
    """

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """PASCAL VOC의 평가 코드를 기반으로 평균 정밀도를 계산합니다.
    이 함수는 평균 정밀도를 계산합니다.
    주어진 정밀도와 회상에서.
    코드는 PASCAL VOC Challenge에서 사용되는 평가 코드를 기반으로합니다.
    Args :
        prec (numpy.array 목록) : 배열 목록입니다.
            : obj :`prec [l]`은 : math :`l` 클래스의 정밀도를 나타냅니다.
            : obj :`prec [l]`이 : obj :`None`이면이 함수는
            : obj :`numpy.nan` for class : math :`l`.
        rec (numpy.array 목록) : 배열 목록입니다.
            : obj :`rec [l]`은 : math :`l` 클래스에 대한 회수를 나타냅니다.
            : obj :`rec [l]`이 : obj :`None`이면이 함수는
            : obj :`numpy.nan` for class : math :`l`.
        use_07_metric (bool) : PASCAL VOC 2007 평가 메트릭 사용 여부
            평균 정밀도를 계산합니다. 기본값은
            : obj :`거짓`.
    보고:
        ~ numpy.ndarray :
        이 함수는 평균 정밀도의 배열을 반환합니다.
        : math :`l`-th 값은 평균 정밀도에 해당합니다.
        클래스 : math :`l`. : obj :`prec [l]`또는 : obj :`rec [l]`이
        : obj :`None`, 해당 값은 : obj :`numpy.nan`으로 설정됩니다.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap