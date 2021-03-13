from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

#
# annType = 'bbox'
# cocoGt = COCO('datasets/instances_val2017.json')
# cocoDt = cocoGt.loadRes('datasets/coco_predictions.json')
# imgIds = sorted(cocoGt.getImgIds())
# # running evaluation
# cocoEval = COCOeval(cocoGt, cocoDt, annType)
# cocoEval.params.imgIds = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

dataDir='./datasets/'
dataType='val2017'
annFile = 'datasets/instances_val2017.json'
cocoGt=COCO(annFile)


cocoDt=cocoGt.loadRes('datasets/coco_predictions.json')

imgIds=sorted(cocoGt.getImgIds())

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
#cocoEval.params.catIds = [1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()