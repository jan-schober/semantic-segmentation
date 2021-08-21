import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv', header = 0)

class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'tl', 'ts', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'ignore']

figure, ax1 = plt.subplots()
for cls in class_names:
    mean_str = "{0:.2f}".format(df[cls].mean())
    str_legend = cls + mean_str
    print(str_legend)
    df[cls].plot(legend = str_legend, linewidth = 0.25)
plt.legend(loc = 'upper right', bbox_to_anchor=(1.0, 1),fontsize='xx-small')
plt.savefig('iou_per_class.png', dpi= 1500)

print(df.mean(axis=0))


print(str(df['road'].mean()))
