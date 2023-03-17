import os

'''
这个是查看四个期像的层数是否一致，不一致则输出该患者的所有期像的层数
'''

# path=r'E:\Bayer\SYS1_PV\Negative_2'
path=r'E:\data\newData\MVI'
path_2=os.listdir(path)
for p1 in path_2:
    a = []
    path_3=os.path.join(path,p1)
    path_4=os.listdir(path_3)
    for p2 in path_4:
        path_5=os.path.join(path_3,p2)
        path_6=os.listdir(path_5)
        path_7=os.path.join(path_5,path_6[0])
        a.append(int(len(os.listdir(path_7))))
    if a[0]!=a[1] or a[0]!=a[2] or a[0]!=a[3]:
        print(p1,a)


