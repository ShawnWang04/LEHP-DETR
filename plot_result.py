import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pwd = os.getcwd()

names = ['YOLOv5m','YOLOv8m','YOLOv9c','RT-DETR-R18','LHEP-DETR(ours)']

plt.figure(figsize=(12, 12), dpi=600)  # Increase figure size and DPI


plt.subplot(2, 2, 1)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['   metrics/precision(B)'] = data['   metrics/precision(B)'].astype(float).replace(np.inf, np.nan)
    data['   metrics/precision(B)'] = data['   metrics/precision(B)'].fillna(data['   metrics/precision(B)'].interpolate())
    plt.plot(data['   metrics/precision(B)'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'precision', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(a)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.subplot(2, 2, 2)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['      metrics/recall(B)'] = data['      metrics/recall(B)'].astype(float).replace(np.inf, np.nan)
    data['      metrics/recall(B)'] = data['      metrics/recall(B)'].fillna(data['      metrics/recall(B)'].interpolate())
    plt.plot(data['      metrics/recall(B)'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'recall', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(b)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.subplot(2, 2, 3)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(float).replace(np.inf, np.nan)
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    plt.plot(data['       metrics/mAP50(B)'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'mAP_0.5', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(c)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.subplot(2, 2, 4)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['    metrics/mAP50-95(B)'] = data['    metrics/mAP50-95(B)'].astype(float).replace(np.inf, np.nan)
    data['    metrics/mAP50-95(B)'] = data['    metrics/mAP50-95(B)'].fillna(data['    metrics/mAP50-95(B)'].interpolate())
    plt.plot(data['    metrics/mAP50-95(B)'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'mAP_0.5:0.95', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(d)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Increase padding

plt.tight_layout(pad=3.0)  # Increase padding around the plots

plt.savefig('metrice_curve.png', dpi=600)  # Increase DPI
print(f'metrice_curve.png saved in {pwd}/metrice_curve.png')

plt.figure(figsize=(15, 10), dpi=1000)  # Increase figure size and DPI

plt.subplot(2, 3, 2)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['         train/cls_loss'] = data['         train/cls_loss'].astype(float).replace(np.inf, np.nan)
    data['         train/cls_loss'] = data['         train/cls_loss'].fillna(data['         train/cls_loss'].interpolate())
    plt.plot(data['         train/cls_loss'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'train/cls_loss', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(a)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.subplot(2, 3, 5)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/db-dota-split/{name}/results.csv')
    data['           val/cls_loss'] = data['           val/cls_loss'].astype(float).replace(np.inf, np.nan)
    data['           val/cls_loss'] = data['           val/cls_loss'].fillna(data['           val/cls_loss'].interpolate())
    plt.plot(data['           val/cls_loss'], label=name, linewidth=2.5)  # Increase line width

    plt.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('epoch')
plt.text(-0.15, 0.5, 'val/cls_loss', ha='center', va='center', rotation='vertical', transform=plt.gca().transAxes, fontsize=14)  # Add and rotate title with increased font size
plt.legend()
plt.text(0.5, -0.2, '(b)', ha='center', transform=plt.gca().transAxes, weight='bold')  # Bold and adjust position

plt.tight_layout(pad=3.0)  # Increase padding around the plots

plt.savefig('loss_curve.png', dpi=600)  # Increase DPI
print(f'loss_curve.png saved in {pwd}/loss_curve.png')
