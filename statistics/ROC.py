from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
#配色方案最终版
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
plt.rc('font', family='Times New Roman')

data = pd.read_excel(r'ROC_External_test_2.xlsx', sheet_name=0, header=None)
train_data = np.array(data)  # np.ndarray()
excel_list = train_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
S1 = excel_listV1[0]
S2 = excel_listV1[1]
S3 = excel_listV1[2]
S4 = excel_listV1[3]

y_true = excel_listV1[4]


fpr1, tpr1, thresholds = metrics.roc_curve(y_true, S1)
roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1
print(roc_auc1)
plt.plot(fpr1, tpr1, '#D6DF7E', label='Strategy1:AUC = %0.2f' % roc_auc1)

fpr2, tpr2, _ = metrics.roc_curve(y_true, S2)
roc_auc2 = metrics.auc(fpr2, tpr2)  # the value of roc_auc1
print(roc_auc2)
plt.plot(fpr2, tpr2, '#187B25', label='Strategy2:AUC = %0.2f' % roc_auc2)

fpr3, tpr3, _ = metrics.roc_curve(y_true, S3)
roc_auc3 = metrics.auc(fpr3, tpr3)  # the value of roc_auc1
print(roc_auc3)
plt.plot(fpr3, tpr3, '#FAA49A', label='Strategy3:AUC = %0.2f' % roc_auc3)

fpr4, tpr4, _ = metrics.roc_curve(y_true, S4)
roc_auc4 = metrics.auc(fpr4, tpr4)  # the value of roc_auc1
print(roc_auc4)
plt.plot(fpr4, tpr4, '#C4391D', label='Strategy4:AUC = %0.2f' % roc_auc4)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0, 1])  # the range of x-axis
# plt.ylim([0, 1])  # the range of y-axis
plt.xlabel('False Positive Rate')  # the name of x-axis
plt.ylabel('True Positive Rate')  # the name of y-axis
plt.title('Receiver operating characteristic in External set')  # the title of figure
plt.savefig("ROC_AUC_External.jpg")
plt.show()
