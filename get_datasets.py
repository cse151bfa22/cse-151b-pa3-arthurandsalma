import csv
import random
from shutil import copyfile
from pycocotools.coco import COCO
from tqdm import tqdm

#make directory and get annotations for training and testing
!mkdir data
!mkdir data/annotations
!cp /datasets/COCO-2015/anno14-15/captions_train2014.json ./data/annotations/
!cp /datasets/COCO-2015/anno14-15/captions_val2014.json ./data/annotations/

!mkdir data/images
!mkdir data/images/train
!mkdir data/images/val
!mkdir data/images/test

coco = COCO('./data/annotations/captions_train2014.json')

#get ids of training images
with open('train_ids.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)
    
trainIds = [int(i) for i in trainIds[0]]

with open('val_ids.csv', 'r') as f:
    reader = csv.reader(f)
    valIds = list(reader)
    
valIds = [int(i) for i in valIds[0]]

for img_id in trainIds:
    path = coco.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/train2014/'+path, './data/images/train/'+path)
for img_id in valIds:
    path = coco.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/train2014/'+path, './data/images/val/'+path)

cocoTest = COCO('./data/annotations/captions_val2014.json')

with open('test_ids.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)
    
testIds = [int(i) for i in testIds[0]]

for img_id in testIds:
    path = cocoTest.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/val2014/'+path, './data/images/test/'+path)

print("done")