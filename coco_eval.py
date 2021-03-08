from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
annType = 'bbox'
cocoGt=COCO('datasets/captions_val2017.json')
cocoDt=cocoGt.loadRes('coco_predictions.json')
imgIds=sorted(cocoGt.getImgIds())
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()