import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

#
if __name__ == '__main__':
    model = RTDETR('runs//weights/best.pt')
    model.val(data='datasets/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # save_json=True, # if you need to cal coco metrice
              project='runs/',
              name='LEHP-DETR',
              )