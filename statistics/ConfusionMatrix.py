import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels_name, title, colorbar=True, cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name,fontsize=12)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,fontsize=12)    # 将标签印在y轴坐标上
    plt.title(title,fontsize=12,weight='bold')    # 图像标题
    plt.ylabel('True label',fontsize=12,weight='bold')
    plt.xlabel('Predicted label',fontsize=12,weight='bold')

data = pd.read_excel(r'External_test_2_S4.xlsx', sheet_name=0, header=None)
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
y_true = excel_listV1[0]
y_pred = excel_listV1[1]
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, ["NCPA", "CCPA", "SAIA","CFPA", "SA", "AN"], "Strategy4 in the External Test Set 2")
plt.savefig("CM-Strategy4.jpg")
plt.show()