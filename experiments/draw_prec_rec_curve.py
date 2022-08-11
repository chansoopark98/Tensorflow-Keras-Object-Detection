import numpy as np
import matplotlib.pyplot as plt

prec = np.load('./prec.npy', allow_pickle=True)
rec = np.load('./rec.npy', allow_pickle=True)

def get_cmap(n, name='tab20'):
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(len(prec))

with open('./pascal_labels.txt') as f:
    CLASSES = f.read().splitlines()
    for i in range(len(prec)):
        plt.plot(rec[i], prec[i], label=str(CLASSES[i]), color=cmap(i))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc=3, ncol=2, fontsize=8)
plt.savefig('./iou_50.png', dpi=100)
plt.show()