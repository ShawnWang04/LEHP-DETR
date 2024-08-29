import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train//weights/best.pt') # select your model.pt path
    model.predict(source='detect_img',
                  project='runs/detect',
                  name='rt-detr',
                  save=True,
                  line_width=1, # 设置为所需的线条宽度
                  conf = 0.5, # 设置为所需的置信度阈值
                  # visualize=True # visualize model features maps
                  )