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


fig = plt.figure(dpi=180, figsize=(18, 12))  # 建立画布，定义画图质量和画布大小
# ax1 = fig.add_subplot(2, 3, 1)  # 构建子图
# ax2 = fig.add_subplot(2, 3, 2)
# ax3 = fig.add_subplot(2, 3, 3)
# ax4 = fig.add_subplot(2, 3, 4)
# ax5 = fig.add_subplot(2, 3, 5)

######################################################################
# labels = ['AP', 'PP', 'HBP', 'T1WI']
# labels = ['AP']
# labels = ['HBP']
# labels = ['PP']
# labels = ['T1WI']
# labels = ['AP_PPatt']
labels = ['fusion_3P']
# labels = ['fusion_4P']
n_time = 2
k_fold = 3

# data1_3=np.load('runs/result_10_4_HBP.npz',allow_pickle=True)
#
# val_AUC1_3=np.round(data1_3['val_acc'],4)
#
# val_fpr1_3=data1_3['val_fpr']
# val_tpr1_3=data1_3['val_tpr']

# AP
max_ = []
max_idx = []
max_idx2 = []
train_max_ = []
trainmax_idx = []
max_AUC_ = []
mean_AUC_ = []
AUC_dix = 0
for i in range(n_time):
    for j in range(k_fold):
        """
        根据ACC来选AUC
        """
        data1 = np.load('runs_5/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[0]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC1 = data1['val_acc']
        val_ACC1 = smooth(val_ACC1)
        # train_ACC1 = data1['train_acc']
        # train_ACC1 = smooth(train_ACC1)
        val_AUC1 = data1['val_AUC']
        val_AUC1 = smooth(val_AUC1)
        max_.append(np.max(val_ACC1))
        max_AUC_.append(np.max(val_AUC1))
        # train_max_.append(np.max(train_ACC1))
        max_idx.append(np.argmax(val_AUC1))
        max_idx2.append(np.argmax(val_ACC1))

# 最大的AUC
AUC_dix = np.argmax(max_AUC_)
# 最大的ACC
AUC_dix2 = np.argmax(max_)
max_id = max_idx[AUC_dix]
max_id2 = max_idx2[AUC_dix2]
# AUC_dix = np.argmax(max_AUC_)
print('val_acc '+str(np.max(max_)))
# print('train_acc '+str(np.max(train_max_)))
print('val_maxAUC '+str(np.max(max_AUC_)))
print('val_meanAUC '+str(np.max(mean_AUC_)))
# fold = int(AUC_dix / n_time)
# time = int(AUC_dix % k_fold)

# 最大的AUC
time1 = int(AUC_dix / k_fold)
fold1 = int(AUC_dix % k_fold)

# 最大的ACC
time2 = int(AUC_dix2 / k_fold)
fold2 = int(AUC_dix2 % k_fold)

# data11 = np.load('runs/result_' + str(1) + '_' + str(4) + '_'+labels[0]+'.npz', allow_pickle=True)
data11 = np.load('runs_5/result_' + str(time1+1) + '_' + str(fold1+1) + '_'+labels[0]+'.npz', allow_pickle=True)



# val_AUC1 = np.round(data11['val_AUC'], 4)[max_id]
# train_ACC1 = np.round(data11['train_acc'], 4)[AUC_dix]
# 最大AUC
x1 = data11['x1'][max_id]
x2 = data11['x2'][max_id]
x3 = data11['x3'][max_id]

x1.cpu().numpy()

plt.plot(x1)
# plt.plot(val_fpr2, val_tpr2, 'b*-', label=labels[1] + ', AUC = ' + str(val_AUC2), linewidth=2.2)
# plt.plot(val_fpr3, val_tpr3, 'g*-', label=labels[2] + ', AUC = ' + str(val_AUC3), linewidth=2.2)
# plt.plot(val_fpr4, val_tpr4, 'k*-', label=labels[3] + ', AUC = ' + str(val_AUC4), linewidth=2.2)

# plt.plot([0, 1], [0, 1], color='navy', linewidth=1.2, linestyle='--') #增加一条AUC=0.5，y=x的虚线
# #
#
# plt.legend(loc='lower right', fontsize=18)
# plt.title('ROC', fontsize=12)
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
#
# plt.tick_params(labelsize=18)
plt.show()


