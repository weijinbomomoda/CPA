#导入相关库
import matplotlib.pyplot as plt    #导入matplotlib的绘图模块

labels = ['L/5', '2L/5', '3L/5', '4L/5', 'L']
x = range(len(labels))

data = pd.read_excel(r'SSL_ACC.xlsx', sheet_name=0, header=None)
SSL_test = np.array(data)  # np.ndarray()
excel_list = SSL_test.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
CPANet = excel_listV1[0]
CPANet_dy = excel_listV1[1]
SSL_CPANet = excel_listV1[2]
SSL_CPANet_dy = excel_listV1[3]

data = pd.read_excel(r'SSL_CK.xlsx', sheet_name=0, header=None)
SSL_test = np.array(data)  # np.ndarray()
excel_list = SSL_test.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
CK = excel_listV1[0]
CK_dy = excel_listV1[1]
SSL_CK = excel_listV1[2]
SSL_CK_dy = excel_listV1[3]

plt.plot(x, SSL_CPANet, marker=None, linestyle='--', color='navy', label='ACC(%) of Semi-Supervised Learning')
plt.errorbar(x, SSL_CPANet, yerr=SSL_CPANet_dy, fmt='bo',ecolor='navy',elinewidth=1,capsize=2)
plt.plot(x, CPANet, marker=None, linestyle='-', color='deepskyblue', label='ACC(%) of Supervised Learning')
plt.errorbar(x, CPANet, yerr=CPANet_dy, fmt='co',ecolor='deepskyblue',elinewidth=1,capsize=2)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'

plt.plot(x, SSL_CK, marker=None, linestyle='--', color='crimson', label='CK(%) of Semi-Supervised Learning')
plt.errorbar(x, SSL_CK, yerr=SSL_CK_dy, fmt='r^',ecolor='crimson',elinewidth=2,capsize=4)
plt.plot(x, CK, marker=None, linestyle='-', color='deeppink', label='CK(%) of Supervised Learning')
plt.errorbar(x, CK, yerr=CK_dy, fmt='m^',ecolor='deeppink',elinewidth=2,capsize=4)


plt.title('Performance comparison in different numbers of labeled images')
plt.xlabel('Numbers of labeled images')
plt.ylabel('(%)')
plt.xticks(x, labels)
plt.yticks(range(50, 105, 10))
plt.legend(loc='lower right')
plt.grid(alpha=1, linestyle='--')
plt.savefig("errostatistic.jpg")
plt.show()
