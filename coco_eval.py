from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
annType = 'bbox'
cocoGt = COCO('datasets/instances_val2017.json')
cocoDt = cocoGt.loadRes('datasets/coco_predictions.json')
imgIds = sorted(cocoGt.getImgIds())
# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()