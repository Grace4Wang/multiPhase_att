import xlwings as xw
import pandas as pd
import numpy as np
from pandas import DataFrame
# labels = ['fusion_3P']
# labels = ['fusion_4P']
labels = ['AP_PPatt']
# labels = ['PP']
# labels = ['HBP']
# labels = ['AP_PP']
# labels = ['T1WI']
n_time = 7
k_fold = 1
midpre_values = []


file = np.load('trainRuns_2P/result_' + str(n_time) + '_' + str(k_fold) + '_'+labels[0]+'.npz', allow_pickle=True)
file.files
tra_trues = file['train_trues']
true_values = tra_trues

tra_preds = file['train_preds']
pre_values = tra_preds

tra_midpreds = file['train_midpre']
midpre_values = np.array(tra_midpreds)

# 保存中间值
writer = pd.ExcelWriter('2Ptrain_midpre.xlsx')
midpre = pd.DataFrame(midpre_values)
midpre.to_excel(writer, 'sheet_1', header='train_midpre', index=False)
# 注意写多个的时候要把所有数据都给了writer后再调用save函数，要不然可能只有一个sheet
writer.save()
writer.close()


# 保存结果
# output_excel = {'tra_trues': [], 'tra_preds': []}
# output_excel['tra_trues'] = tra_trues
# output_excel['tra_preds'] = tra_preds
# outputEx = pd.DataFrame(output_excel)
# outputEx.to_excel('2Ptrain.xlsx', index=False)


print('finish')