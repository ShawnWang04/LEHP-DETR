import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/LEHP-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=380,
                batch=16,
                workers=8,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )