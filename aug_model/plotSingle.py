import numpy as np
import matplotlib.pyplot as plt

'''
这是画ROC的代码
'''


def smooth(scalar, weight=0.5):
    '''
    平滑函数，平滑系数weight可以适当增大
    '''
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


fig = plt.figure(dpi=300, figsize=(18, 12))  # 建立画布，定义画图质量和画布大小
# ax1 = fig.add_subplot(2, 3, 1)  # 构建子图
# ax2 = fig.add_subplot(2, 3, 2)
# ax3 = fig.add_subplot(2, 3, 3)
# ax4 = fig.add_subplot(2, 3, 4)
# ax5 = fig.add_subplot(2, 3, 5)

######################################################################
labels = ['AP', 'PP', 'HBP', 'T1WI']
# labels = ['fusion']
# labels = ['HBP']
k_fold = 5
n_time = 2

max_ = []
max_idx = []
for i in range(n_time):
    for j in range(k_fold):
        """
        根据ACC来选AUC
        """
        data1 = np.load('runs/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[0]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC1 = data1['val_acc']
        val_ACC1 = smooth(val_ACC1)
        max_idx.append(np.argmax(val_ACC1))
        max_.append(np.max(val_ACC1))

AUC_dix = np.argmax(max_)
fold = int(AUC_dix / n_time)
time = int(AUC_dix % k_fold)

data11 = np.load('runs/result_' + str(fold+1) + '_' + str(time+1) + '_'+labels[0]+'.npz', allow_pickle=True)

val_AUC1 = np.round(data11['val_AUC'], 4)[AUC_dix]
val_fpr1 = data11['val_fpr'][AUC_dix]
val_tpr1 = data11['val_tpr'][AUC_dix]

max_ = []
max_idx = []
for i in range(n_time):
    for j in range(k_fold):
        data2 = np.load('runs/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[1]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC2 = data2['val_acc']
        val_ACC2 = smooth(val_ACC2)
        max_idx.append(np.argmax(val_ACC2))
        max_.append(np.max(val_ACC2))
AUC_dix = np.argmax(max_)
fold = int(AUC_dix / n_time)
time = int(AUC_dix % k_fold)
#
data22 = np.load('runs/result_' + str(fold+1) + '_' + str(time+1) + '_'+labels[1]+'.npz', allow_pickle=True)
val_AUC2 = np.round(data22['val_AUC'], 4)[AUC_dix]
val_fpr2 = data22['val_fpr'][AUC_dix]
val_tpr2 = data22['val_tpr'][AUC_dix]
#
max_ = []
max_idx = []
for i in range(n_time):
    for j in range(k_fold):
        data3 = np.load('runs/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[2]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC3 = data3['val_acc']
        val_ACC3 = smooth(val_ACC3)
        max_idx.append(np.argmax(val_ACC3))
        max_.append(np.max(val_ACC3))
AUC_dix = np.argmax(max_)
fold = int(AUC_dix / n_time)
time = int(AUC_dix % k_fold)
#
data33 = np.load('runs/result_' + str(fold+1) + '_' + str(time+1) + '_'+labels[2]+'.npz', allow_pickle=True)
val_AUC3 = np.round(data33['val_AUC'], 4)[AUC_dix]
val_fpr3 = data33['val_fpr'][AUC_dix]
val_tpr3 = data33['val_tpr'][AUC_dix]

max_ = []
max_idx = []
for i in range(n_time):
    for j in range(k_fold):
        data4 = np.load('runs/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[3]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC4 = data4['val_acc']
        val_ACC4 = smooth(val_ACC4)
        max_idx.append(np.argmax(val_ACC4))
        max_.append(np.max(val_ACC4))
AUC_dix = np.argmax(max_)
fold = int(AUC_dix / n_time)
time = int(AUC_dix % k_fold)

data44 = np.load('runs/result_' + str(fold+1) + '_' + str(time+1) + '_'+labels[3]+'.npz', allow_pickle=True)
val_AUC4 = np.round(data44['val_AUC'], 4)[AUC_dix]
val_fpr4 = data44['val_fpr'][AUC_dix]
val_tpr4 = data44['val_tpr'][AUC_dix]

# ########################################################

# ax1.plot(val_fpr1, val_tpr1, 'r*-', label=labels[0] + ', AUC = ' + str(val_AUC1), linewidth=2.2)#子图的写法
# ax2.plot(val_fpr2, val_tpr2, 'r*-', label=labels[1] + ', AUC = ' + str(val_AUC1), linewidth=2.2)#子图的写法

plt.plot(val_fpr1, val_tpr1, 'r*-', label=labels[0] + ', AUC = ' + str(val_AUC1), linewidth=2.2)
plt.plot(val_fpr2, val_tpr2, 'b*-', label=labels[1] + ', AUC = ' + str(val_AUC2), linewidth=2.2)
plt.plot(val_fpr3, val_tpr3, 'g*-', label=labels[2] + ', AUC = ' + str(val_AUC3), linewidth=2.2)
plt.plot(val_fpr4, val_tpr4, 'k*-', label=labels[3] + ', AUC = ' + str(val_AUC4), linewidth=2.2)

plt.plot([0, 1], [0, 1], color='navy', linewidth=1.2, linestyle='--') #增加一条AUC=0.5，y=x的虚线
#

plt.legend(loc='lower right', fontsize=18)
plt.title('ROC', fontsize=18)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)

plt.tick_params(labelsize=18)
plt.show()