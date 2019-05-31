#!/usr/bin/python
from pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import sys

# create coco object and cocoRes object
coco = COCO(sys.argv[1])
cocoRes = coco.loadRes(sys.argv[2])
# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)
# evaluate on a subset of images by setting
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()
# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
for key,value in cocoEval.imgToEval.iteritems():
    print(key,value)
