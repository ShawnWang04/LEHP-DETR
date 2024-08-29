import matplotlib.pyplot as plt

# 模型参数量和准确率数据
params = [25.1, 25.8, 25.3, 19.9, 6.5]
acc = [0.706, 0.719, 0.720, 0.718, 0.701]
models = ['YOLOv5m','YOLOv8m','YOLOv9c','RT-DETR-R18','LHEP-DETR(ours)']
colors = ['b', 'g', 'r', 'c', 'm', 'brown', 'orange', 'black', 'pink']

lmodels = ['YOLOv5m','YOLOv8m','YOLOv9c','RT-DETR-R18','LHEP-DETR(ours)']
lcolors = ['b', 'g', 'r', 'c', 'm', 'pink', 'black', 'orange', 'brown']

# 设置图中圆圈地大小
point_size = [p * 40 for p in params]

# 指定画布大小
plt.figure(figsize=(8, 6), dpi=1000)

# 绘制散点图,不显示标签
for i in range(len(models)):
    plt.scatter(params[i], acc[i], s=point_size[i], alpha=1, color=colors[i])
    # if i == 0:
    #     plt.annotate(models[i], xy=(params[i], acc[i]), xytext=(15, -5), textcoords='offset points', weight='bold',
    #                  family='Times New Roman', fontsize=8)
    # elif i == 1:
    #     plt.annotate(models[i], xy=(params[i], acc[i]), xytext=(20, -5), textcoords='offset points', weight='bold',
    #                  family='Times New Roman', fontsize=8)
    # elif i == 4:
    #     plt.annotate(models[i], xy=(params[i], acc[i]), xytext=(5, -5), textcoords='offset points', weight='bold',
    #                  family='Times New Roman', fontsize=8)
    # elif i == 7:
    #     plt.annotate(models[i], xy=(params[i], acc[i]), xytext=(-35, 8), textcoords='offset points', weight='bold',
    #                  family='Times New Roman', fontsize=8)
    # else:
    #     plt.annotate(models[i], xy=(params[i], acc[i]), xytext=(10, -5), textcoords='offset points', weight='bold',
    #                  family='Times New Roman', fontsize=8)

# 绘制虚拟点
for i in range(len(lmodels)):
    plt.plot([], [], 'o', markersize=6, alpha=1, color=lcolors[i], label=lmodels[i])

# 横坐标轴范围
plt.xlim(0, 50)

# 纵坐标轴范围
plt.ylim(0.5, 1)

ax = plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)  # 设置右边坐标轴的粗细

# 设置横纵坐标标签格式
plt.xticks(fontsize=14, family='Times New Roman', weight='bold')
plt.yticks(fontsize=14, family='Times New Roman', weight='bold')

# 添加横纵坐标标题格式
plt.xlabel('Params/M', fontsize=16, family='Times New Roman', weight='bold')
plt.ylabel('Precision', fontsize=16, family='Times New Roman', weight='bold')

# 添加图例并调整位置,设置字体为Times New Roman
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.0), scatterpoints=1,
           prop={'family': 'Times New Roman', 'size': 12}, frameon=False, labelspacing=0.2)

# 存储图像并调整分辨率
plt.savefig('accure-Params.png', format='png', transparent=True, bbox_inches='tight', dpi=1000)
# 显示图像
plt.tight_layout()
plt.show()
