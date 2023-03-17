import os
from shutil import  copy,move
from tqdm import tqdm


'''
这个代码的功能是将每个患者中的HBP文件内的勾画’.nii‘提取成单独文件

这是旧版的，新版已在服务器中更改
新版旧版主要区别是新来的一批数据的文件夹分布和旧的不一致，需要进行判别并更改文件读取路径
'''

def move_nii(path,save_path):
    path_1=os.listdir(path)
    for p1 in tqdm(path_1):
        path_2=os.path.join(path,p1)
        path_3=os.listdir(path_2)
        for p2 in path_3:
            if 'HBP' in p2:
                path_4=os.path.join(path_2,p2)
                path_4_1=os.listdir(path_4)
                path_4_2=os.path.join(path_4,path_4_1[0])
                path_5=os.listdir(path_4_2)
                for p3 in path_5:
                    if '.nii' in p3:
                        path_6=os.path.join(path_4_2,p3)
                        move(path_6,save_path)
                        break
                break

save_path=r'E:\LSX\Bayer\SYS1_PV\nii_3'
if os.path.exists(save_path)==False:
    os.makedirs(save_path)

# path=r'E:\Bayer\SYS1_PV\Negative_2'
path=r'E:\data\newData\MVI'

move_nii(path,save_path)
