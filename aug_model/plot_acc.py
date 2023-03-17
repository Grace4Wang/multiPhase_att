import numpy as np
import matplotlib.pyplot as plt

'''
这是画ACC的代码
'''

# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times

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


# fig = plt.figure(dpi=300, figsize=(18, 12))  # 建立画布，定义画图质量和画布大小
# ax1 = fig.add_subplot(2, 3, 1)  # 构建子图
# ax2 = fig.add_subplot(2, 3, 2)
# ax3 = fig.add_subplot(2, 3, 3)
# ax4 = fig.add_subplot(2, 3, 4)
# ax5 = fig.add_subplot(2, 3, 5)

######################################################################
# labels = ['AP', 'PP', 'HBP', 'T1WI']
labels = ['fusion_3P']
# labels = ['AP']
# labels = ['AP_PP']
n_time = 6
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
train_max_ = []
trainmax_idx = []
mean_acc_ = []
std_ = []
sensitivity_ = []
specificity_ = []
for i in range(n_time):
    for j in range(k_fold):
        """
        绘制ACC和LOSS
        """
        data1 = np.load('runs_3/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[0]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC1 = data1['val_acc']
        val_loss = data1['val_loss']
        train_acc = data1['train_acc']
        train_loss = data1['train_loss']
        # sensitivity = data1['sensitivity']
        # specificity = data1['specificity']
        val_ACC1 = smooth(val_ACC1)

        mean_acc_.append(np.mean(val_ACC1))
        std_.append(np.std(val_ACC1))
        # sensitivity_.append(np.mean(sensitivity))
        # specificity_.append(np.mean(specificity))

        max_idx.append(np.argmax(val_ACC1))
        max_.append(np.max(val_ACC1))

AUC_dix = np.argmax(max_)
print(np.mean(max_))
time = int(AUC_dix / k_fold)
fold = int(AUC_dix % k_fold)

print(np.mean(mean_acc_))
print(np.mean(std_))
print(np.std(max_))
print("sensitivity: " + str(np.mean(sensitivity_)))
print("specificity: " + str(np.mean(specificity_)))

print('time ' + str(time + 1))
print('fold ' + str(fold + 1))
# data11 = np.load('runs/result_' + str(time+1) + '_' + str(fold+1) + '_'+labels[0]+'.npz', allow_pickle=True)

data11 = np.load('runs_3/result_' + str(n_time) + '_' + str(k_fold) + '_'+labels[0]+'.npz', allow_pickle=True)

# val_AUC1 = np.round(data11['val_AUC'], 4)[AUC_dix]
# train_ACC1 = np.round(data11['train_acc'], 4)[AUC_dix]
# val_fpr1 = data11['val_fpr'][AUC_dix]
# val_tpr1 = data11['val_tpr'][AUC_dix]

# print(np.max(val_AUC1))


iter_num = []
iter_num = train_loss[0:-1:100]

x_train_loss = range(len(iter_num))
x_train_acc = multiple_equal(x_train_loss, range(len(iter_num)))


# ########################################################

# ax1.plot(val_fpr1, val_tpr1, 'r*-', label=labels[0] + ', AUC = ' + str(val_AUC1), linewidth=2.2)#子图的写法
# ax2.plot(val_fpr2, val_tpr2, 'r*-', label=labels[1] + ', AUC = ' + str(val_AUC1), linewidth=2.2)#子图的写法
# plt.ylabel('accuracy')
fig = plt.figure(dpi=200,figsize=(10,8))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# ax1.xlabel('iter')
# ax1.ylabel('losses')
ax1.plot(train_loss, 'b-', label=labels[0] + '_train_loss')
ax1.legend(loc='upper right',fontsize=10)
ax1.set_title('train_loss',fontsize=10)
ax1.set_xlabel('iter',fontsize=10)
ax1.set_ylabel('loss',fontsize=10)


# ax3.xlabel('iter')
# ax3.ylabel('acc')
ax2.plot(train_acc, 'r-', label=labels[0]+'_trian_acc')
ax2.legend(loc='lower right',fontsize=10)
ax2.set_title('train_acc',fontsize=10)
ax2.set_xlabel('iter',fontsize=10)
ax2.set_ylabel('acc',fontsize=10)

ax3.plot(val_loss, 'b--', label='val_loss')
ax3.legend(loc='lower right',fontsize=10)
ax3.set_title('val_loss',fontsize=10)
ax3.set_xlabel('iter',fontsize=10)
ax3.set_ylabel('loss',fontsize=10)

ax4.plot(val_ACC1, 'r--', label='val_acc')
ax4.legend(loc='lower right',fontsize=10)
ax4.set_xlabel('iter',fontsize=10)
ax4.set_ylabel('acc',fontsize=10)
plt.legend()
plt.show()


# plt.plot(train_loss, 'r--', label="val_loss")
# plt.plot(val_fpr2, val_tpr2, 'b*-', label=labels[1] + ', AUC = ' + str(val_AUC2), linewidth=2.2)
# plt.plot(val_fpr3, val_tpr3, 'g*-', label=labels[2] + ', AUC = ' + str(val_AUC3), linewidth=2.2)
# plt.plot(val_fpr4, val_tpr4, 'k*-', label=labels[3] + ', AUC = ' + str(val_AUC4), linewidth=2.2)

# plt.plot([0, 1], [0, 1], color='navy', linewidth=1.2, linestyle='--') #增加一条AUC=0.5，y=x的虚线
#

# plt.legend(loc='lower right', fontsize=18)
# plt.title('ACC&Loss', fontsize=18)
# plt.xlabel('iteration', fontsize=18)
# # plt.ylabel('True Positive Rate', fontsize=18)
#
# plt.tick_params(labelsize=18)
# plt.show()


