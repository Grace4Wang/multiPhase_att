import xlwings as xw
import pandas as pd
import numpy as np
from pandas import DataFrame
labels = ['fusion_3P']
# labels = ['AP_PP']
# labels = ['AP_PPatt']
# labels = ['PP']
# labels = ['HBP']
# labels = ['T1WI']
n_time = 2
k_fold = 2
midpre_values = []


file = np.load('runs_5/result_' + str(n_time) + '_' + str(k_fold) + '_'+labels[0]+'.npz', allow_pickle=True)
file.files
# val_trues = file['val_trues']
# val_values = np.array(val_trues)
#
# val_preds = file['val_preds']
# pre_values = np.array(val_preds)
#
# val_midpreds = file['val_midpre']
# midpre_values = val_midpreds

x1 = file['x1']
x2 = file['x2']
x3 = file['x3']
# x1 = val_midpreds


# 保存中间值
# data = np.zeros((1344, 42, 2))
# data = midpre_values
# writer = pd.ExcelWriter('2Pval_midpre.xlsx', engine='xlsxwriter')
# for i in range(0, 42):
#     df = pd.DataFrame(data[:,i,:])
#     df.to_excel(writer, sheet_name='bin%d' % i)
#
# writer.save()
# writer.close()


# 保存结果
# writer = pd.ExcelWriter('2Pval.xlsx')
# val_tr = pd.DataFrame(val_values)
# val_pre = pd.DataFrame(pre_values)
# val_tr.to_excel(writer, 'val_trues', float_format='%.2f', header=False, index=False)
# val_pre.to_excel(writer, 'val_preds', header=False, index=False)
# writer.save()
# writer.close()

writer = pd.ExcelWriter('3Peach.xlsx')
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
x3 = pd.DataFrame(x3)
x1.to_excel(writer, 'x1', header=False, index=False)
x2.to_excel(writer, 'x2', header=False, index=False)
x3.to_excel(writer, 'x3', header=False, index=False)
writer.save()
writer.close()

print('finish')
