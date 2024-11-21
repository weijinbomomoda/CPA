import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决图像中的'-'负号的乱码问题
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(r'ACC.xlsx', sheet_name=0, header=None)
test_data = np.array(data)  # np.ndarray()
excel_list = test_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
A_ACC = excel_listV1[0]
B_ACC = excel_listV1[1]
C_ACC = excel_listV1[2]
D_ACC = excel_listV1[3]
E_ACC = excel_listV1[4]
F_ACC = excel_listV1[5]

data = pd.read_excel(r'CK.xlsx', sheet_name=0, header=None)
test_data = np.array(data)  # np.ndarray()
excel_list = test_data.tolist()  # list
excel_listV1 = list(map(list, zip(*excel_list)))
A_CK = excel_listV1[0]
B_CK = excel_listV1[1]
C_CK = excel_listV1[2]
D_CK = excel_listV1[3]
E_CK = excel_listV1[4]
F_CK = excel_listV1[5]


fig = plt.figure()
ax = fig.add_subplot()

# 每个刻度标签下有几个group就有几个箱子
group_dataA = [A_ACC, A_CK]

boxplot_dataABC_A = [A_ACC, B_ACC, C_ACC, D_ACC, E_ACC, F_ACC]
boxplot_dataABC_B = [A_CK, B_CK, C_CK, D_CK, E_CK, F_CK]

'''
# 橙绿蓝
color_list = ['powderblue', 'pink', 'lightgreen', 'blanchedalmond', 'lightcoral']
medianprops_color = ['dodgerblue', 'palevioletred', 'g', 'darkorange', 'r']
scatter_color = ['navy', 'crimson', 'darkgreen', 'darkgoldenrod', 'darkred']
'''
color_list = ['powderblue', 'pink']
medianprops_color = ['dodgerblue', 'palevioletred']
scatter_color = ['navy', 'crimson']


x_labels = ['ViT', 'ResNet152', 'ResNet50', 'ResNet50_SE', 'ResNet50_CBAM', 'Ours Model']
legend_labels = ['ACC', 'CK']
length = len(x_labels)
x_loc = np.arange(length)

group_number = len(group_dataA)
total_width = 0.6
box_total_width = total_width * 0.65
interval_total_width = total_width * 0.05
box_width = box_total_width / group_number

###################################################
if group_number == 1:
    interval_width = interval_total_width
else:
    interval_width = interval_total_width / (group_number - 1)

###################################################
if group_number % 2 == 0:
    x1_box = x_loc - (group_number / 2 - 1) * box_width - box_width / 2 - (group_number / 2 - 1) * interval_width - interval_width / 2
else:
    x1_box = x_loc - ((group_number - 1) / 2) * box_width - ((group_number - 1) / 2) * interval_width
x_list_box = [x1_box + box_width * i + interval_width * i for i in range(group_number)]


boxplot_data = [boxplot_dataABC_A, boxplot_dataABC_B]

for i in range(len(boxplot_data)):
    #####################################################################
    # 先画boxplot
    #######################
    # boxplot_data_num用来统计每组数据的长度, 画scatter图时会用到
    boxplot_data_num = []
    for j in boxplot_data[i]:
        boxplot_data_num_tmp = len(j)
        boxplot_data_num.append(boxplot_data_num_tmp)
    #######################
    ax.boxplot(boxplot_data[i], positions=x_list_box[i], widths=box_width, patch_artist=True,
               medianprops={'lw': 1, 'color': medianprops_color[i]},
               boxprops={'facecolor': color_list[i], 'edgecolor': 'black'},
               capprops={'lw': 1, 'color': 'black'},
               whiskerprops={'ls': '-', 'lw': 1, 'color': 'black'},
               showfliers=True, zorder=1)
    # flierprops = {'marker': 'o', 'markerfacecolor': color_list[i], 'markeredgecolor': color_list[i], 'markersize': 8}
    #####################################################################
    # 再画scatter
    # 将每一组箱线图统计的所有点绘制在图上
    # spotx是每一组箱线图所有的点的横坐标

    spotx = []
    for j_spotx, k_spotx in zip(x_list_box[i], boxplot_data_num):
        spotx_tmp = [j_spotx] * k_spotx
        spotx.append(spotx_tmp)
    # print('$$$spotx:', spotx)
    ax.scatter(spotx, boxplot_data[i], c=scatter_color[i], s=10, label=legend_labels[i], zorder=2)

ax.grid(True, ls=':', color='b', alpha=0.3)
plt.title('Performance comparison of Models ', fontweight='bold')
ax.set_xticks(x_loc)
ax.set_xticklabels(x_labels,rotation=45)
ax.set_ylabel('（%）', fontweight='bold')
################################################################################################################
################################################################################################################
'''
plt.legend(title='Performance CK', loc='center left', bbox_to_anchor=(1.02, 0.5), facecolor='None', edgecolor='#000000',
           frameon=True, ncol=1, markerscale=3, borderaxespad=0, handletextpad=0.1, fontsize='8', title_fontsize='8')
'''

plt.legend(title='Performance CK', loc='lower right')
################################################################################################################
################################################################################################################
plt.xticks(weight='bold')
plt.yticks(weight='bold')
fig.tight_layout()
plt.savefig("ACC & CK comparison of Models.jpg")
plt.show()