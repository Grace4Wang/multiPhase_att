import pandas as pd
import os
import random

'''
这个代码的主要功能是将excel中的label用txt来保存， 保存的形式为 ’患者名（文件名）+空格+label‘
'''

excel_path = r'F:\FTPfiles\SYS1_PV\CK19label.xlsx'  # excel 路径
data_path = r'F:\FTPfiles\SYS1_PV\CK19'  # 数据路径

df = pd.read_excel(excel_path)  # 读excel
patient_id = df["patient_id"].values  # 读patient_id表格数据
# CK19 = df["CK19"].values  # 读CK19列数据
# MVI = df["MVI"].values  # 读MVI列数据
CK19 = df["CK19"].values # 读CK19列数据

patient_id_2 = os.listdir(data_path)  # 根据数据路径获取患者id

# high_CK19, low_CK19, high_mvi, low_mvi = [], [], [], []
# high_mvi, low_mvi = [], []
high_CK19, low_CK19 = [], []

with open("./txt_ck19/label.txt", "w") as f:
    '''
    制作总的label.txt 
    保存的形式为 ’患者名（文件名）+空格+CK19标签+空格+MVI标签‘
    '''
    for i in range(len(patient_id_2)):
        # if CK19[i] == 0:
        #     low_CK19.append(patient_id_2[i][0:3])
        # else:
        #     high_CK19.append(patient_id_2[i][0:3])

        if CK19[i] == 0:
            low_CK19.append(patient_id_2[i][0:3])
        else:
            high_CK19.append(patient_id_2[i][0:3])

        # out = str(patient_id_2[i][0:3]) + ' ' + str(CK19[i]) + ' ' + str(MVI[i])
        out = str(patient_id_2[i][0:3])  + ' ' + str(CK19[i])
        f.write(out + '\n')

# print(len(high_CK19), len(low_CK19), len(high_mvi), len(low_mvi))
print(len(high_CK19), len(low_CK19))
# len(high_mvi),len(low_mvi))=70，140

# fold_high = [0, 5, 10, 15, 20]  # 根据想要的折数和验证集的大小，并尽可能正负样本均衡来写这个的数值
# fold_low = [0, 8, 16, 24, 32]

# fold_high = [0, 14, 28, 42, 56]  # 根据想要的折数和验证集的大小，并尽可能正负样本均衡来写这个的数值
# fold_low = [0, 28, 56, 84, 112]

fold_high = [0, 8, 16, 24, 32, 40]
fold_low = [0, 23, 45, 68, 90, 113]

with open('txt_ck19/val_1.txt', 'w') as val_1f:
    '''
    制作mvi验证集val.txt  相似的制作ck19的val.txt
    保存的形式为 ’患者名（文件名）+空格+MVI标签‘
    '''
    high_ck19_val_1 = high_CK19[fold_high[0]:fold_high[1]]
    low_ck19_val_1 = low_CK19[fold_low[0]:fold_low[1]]
    for i in high_ck19_val_1:
        # val_1f.write(str(i) + '_ori 1 ' + '\n')
        val_1f.write(str(i) + ' 1 ' + '\n')
    for i in low_ck19_val_1:
        # val_1f.write(str(i) + '_ori 0 ' + '\n')
        val_1f.write(str(i) + ' 0 ' + '\n')

with open('txt_ck19/val_2.txt', 'w') as val_2f:
    high_ck19_val_2 = high_CK19[fold_high[1]:fold_high[2]]
    low_ck19_val_2 = low_CK19[fold_low[1]:fold_low[2]]
    for i in high_ck19_val_2:
        val_2f.write(str(i) + ' 1 ' + '\n')
    for i in low_ck19_val_2:
        val_2f.write(str(i) + ' 0 ' + '\n')

with open('txt_ck19/val_3.txt', 'w') as val_3f:
    high_ck19_val_3 = high_CK19[fold_high[2]:fold_high[3]]
    low_ck19_val_3 = low_CK19[fold_low[2]:fold_low[3]]
    for i in high_ck19_val_3:
        val_3f.write(str(i) + ' 1 ' + '\n')
    for i in low_ck19_val_3:
        val_3f.write(str(i) + ' 0 ' + '\n')

with open('txt_ck19/val_4.txt', 'w') as val_4f:
    high_ck19_val_4 = high_CK19[fold_high[3]:fold_high[4]]
    low_ck19_val_4 = low_CK19[fold_low[3]:fold_low[4]]
    for i in high_ck19_val_4:
        val_4f.write(str(i) + ' 1 ' + '\n')
    for i in low_ck19_val_4:
        val_4f.write(str(i) + ' 0 ' + '\n')

with open('txt_ck19/val_5.txt', 'w') as val_5f:
    high_ck19_val_5 = high_CK19[fold_high[4]:]
    low_ck19_val_5 = low_CK19[fold_low[4]:]
    for i in high_ck19_val_5:
        val_5f.write(str(i) + ' 1 ' + '\n')
    for i in low_ck19_val_5:
        val_5f.write(str(i) + ' 0 ' + '\n')

with open('txt_ck19/train_1.txt', 'w') as train_1f:
    '''
    制作训练集train.txt 
    保存的形式为 ’患者名（文件名）+下划线+i_j_k(增强时滑动的数值)+空格+MVI标签‘
    '''
    high_ck19_train_1 = high_CK19[fold_high[1]:]
    low_ck19_train_1 = low_CK19[fold_low[1]:]
    i = 0
    while i < 16:
        j = 0
        while j < 16:
            k = 0
            while k < 16:

                for m in high_ck19_train_1:
                    train_1f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 1 ' + '\n')
                for m in low_ck19_train_1:
                    train_1f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 0 ' + '\n')

                k = k + 2   #滑动步长
            j = j + 2
        i = i + 2

with open('txt_ck19/train_2.txt', 'w') as train_2f:
    high_ck19_train_2 = high_CK19[fold_high[0]:fold_high[1]] + high_CK19[fold_high[2]:]
    low_ck19_train_2 = low_CK19[fold_low[0]:fold_low[1]] + low_CK19[fold_low[2]:]
    i = 0
    while i < 16:
        j = 0
        while j < 16:
            k = 0
            while k < 16:

                for m in high_ck19_train_2:
                    train_2f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 1 ' + '\n')
                for m in low_ck19_train_2:
                    train_2f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 0 ' + '\n')
                k = k + 2
            j = j + 2
        i = i + 2

with open('txt_ck19/train_3.txt', 'w') as train_3f:
    high_ck19_train_3 = high_CK19[fold_high[0]:fold_high[2]] + high_CK19[fold_high[3]:]
    low_ck19_train_3 = low_CK19[fold_low[0]:fold_low[2]] + low_CK19[fold_low[3]:]
    i = 0
    while i < 16:
        j = 0
        while j < 16:
            k = 0
            while k < 16:

                for m in high_ck19_train_3:
                    train_3f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 1 ' + '\n')
                for m in low_ck19_train_3:
                    train_3f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 0 ' + '\n')
                k = k + 2
            j = j + 2
        i = i + 2

with open('txt_ck19/train_4.txt', 'w') as train_4f:
    high_ck19_train_4 = high_CK19[fold_high[0]:fold_high[3]] + high_CK19[fold_high[4]:]
    low_ck19_train_4 = low_CK19[fold_low[0]:fold_low[3]] + low_CK19[fold_low[4]:]
    i = 0
    while i < 16:
        j = 0
        while j < 16:
            k = 0
            while k < 16:

                for m in high_ck19_train_4:
                    train_4f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 1 ' + '\n')
                for m in low_ck19_train_4:
                    train_4f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 0 ' + '\n')
                k = k + 2
            j = j + 2
        i = i + 2

with open('txt_ck19/train_5.txt', 'w') as train_5f:
    high_ck19_train_5 = high_CK19[fold_high[0]:fold_high[4]]
    low_ck19_train_5 = low_CK19[fold_low[0]:fold_low[4]]
    i = 0
    while i < 16:
        j = 0
        while j < 16:
            k = 0
            while k < 16:

                for m in high_ck19_train_5:
                    train_5f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 1 ' + '\n')
                for m in low_ck19_train_5:
                    train_5f.write(str(m) + '_' + str(i) + '_' + str(j) + '_' + str(k) + ' 0 ' + '\n')
                k = k + 2
            j = j + 2
        i = i + 2

