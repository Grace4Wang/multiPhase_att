import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
输出最好一折最好一次的结果
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


######################################################################
# labels = ['AP', 'PP', 'HBP', 'T1WI']
labels = ['AP_PPatt']
# labels = ['HBP']
# labels = ['PP']
# labels = ['T1WI']
# labels = ['AP_PP']
# labels = ['fusion_3P']
# labels = ['fusion_4P']
n_time = 7
k_fold = 1


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
        data1 = np.load('temps/result_' + str(i+1) + '_' + str(j+1) + '_'+labels[0]+'.npz', allow_pickle=True)  # 读取保存的npz数据
        val_ACC1 = data1['val_acc']
        val_ACC1 = smooth(val_ACC1)
        # train_ACC1 = data1['train_acc']
        # train_ACC1 = smooth(train_ACC1)
        val_AUC1 = data1['val_AUC']
        val_AUC1 = smooth(val_AUC1)
        max_.append(np.max(val_ACC1))
        max_AUC_.append(np.max(val_AUC1))
        mean_AUC_.append(np.mean(val_AUC1))
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
data11 = np.load('temps/result_' + str(time1+1) + '_' + str(fold1+1) + '_'+labels[0]+'.npz', allow_pickle=True)

data12 = np.load('temps/result_' + str(time2+1) + '_' + str(fold2+1) + '_'+labels[0]+'.npz', allow_pickle=True)



# val_AUC1 = np.round(data11['val_AUC'], 4)[max_id]
# train_ACC1 = np.round(data11['train_acc'], 4)[AUC_dix]
# 最大AUC
val_AUC1 = np.round(data11['val_AUC'], 4)[max_id]
val_fpr1 = data11['val_fpr'][max_id]
val_tpr1 = data11['val_tpr'][max_id]
# 最大ACC取的AUC
val_AUC2 = np.round(data12['val_AUC'], 4)[max_id2]
val_fpr2 = data12['val_fpr'][max_id2]
val_tpr2 = data12['val_tpr'][max_id2]

val_maxAUC = np.round(max_AUC_,4)
val_maxAUC = np.max(val_maxAUC)

best_AUC = val_AUC1
if val_AUC2 > best_AUC:
    best_AUC = val_AUC2
    val_fpr1 = val_fpr2
    val_tpr1 = val_tpr2
    print('time2 ' + str(time2 + 1))
    print('fold2 ' + str(fold2 + 1))
else:
    print('time1 ' + str(time1 + 1))
    print('fold1 ' + str(fold1 + 1))

print(np.max(val_AUC1))
print(np.max(val_AUC2))

val_trues = data11['val_trues']
val_values = np.array(val_trues)[max_id]

val_preds = data11['val_preds']
pre_values = np.array(val_preds)[max_id]
print(pre_values)

output_excel = {'val_trues': [], 'val_preds': []}
output_excel['val_trues'] = val_values
output_excel['val_preds'] = pre_values
outputEx = pd.DataFrame(output_excel)
outputEx.to_excel('2Pval_bestTime.xlsx', index=False)

# val_midpreds = data11['val_midpre']
# midpre_values = val_midpreds


