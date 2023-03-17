import torch
from torch.utils.data import Dataset
import SimpleITK
import numpy as np
import os


'''
这个代码的主要目的是读取txt，根据路劲+txt中的患者编号读取患者数据并输出，并同时根据txt中的label输出标签
目前存在可能影响性能的bug
'''
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, data_dir1, data_dir2, txt):
        super(MyDataSet, self).__init__()

        data = []
        fh = open(txt, 'r')    # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  #按行循环txt文本中的内容
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，即删除换行符
            words = line.split(' ')  # 按空格来切分内容
            data.append((words[0], int(words[1])))  #保存患者名及其标签标签

        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        # self.data_dir4 = data_dir4
        

        self.data = data

    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        fn, label = self.data[index]
        
        image_path1 = os.path.join(self.data_dir1, fn)
        img1 = SimpleITK.ReadImage(image_path1)
        img1 = SimpleITK.GetArrayFromImage(img1)
        img1 = np.array(img1, dtype='float32')
        image1=img1[np.newaxis,:]     #给数据增加一个值为1的channel维度

        image_path2 = os.path.join(self.data_dir2, fn)
        img2 = SimpleITK.ReadImage(image_path2)
        img2 = SimpleITK.GetArrayFromImage(img2)
        img2 = np.array(img2, dtype='float32')
        image2 = img2[np.newaxis, :]

        # image_path4 = os.path.join(self.data_dir4, fn)
        # img4 = SimpleITK.ReadImage(image_path4)
        # img4 = SimpleITK.GetArrayFromImage(img4)
        # img4 = np.array(img4, dtype='float32')
        # image4 = img4[np.newaxis, :]
        
        '''
        性能不好原因可能出现在这里，
        2D图像做读取时，用Image.open（）或cv2.read（）来读取会读出原图的channel，RGB为3，灰度图为1.
        而我们的数据为3D数据，应该有对应的3D读取方式，我这里用SimpleITK读出来的没有channel维度，故直接增加了一个chennel维度以适应模型
        具体可以参考网上的医学图像3D分割算法，比如3DUnet是如何读取3D的医学图像的，以及是如何输入进网络的。
        '''

        data1=torch.from_numpy(image1)
        data2=torch.from_numpy(image2)
        # data4=torch.from_numpy(image4)#转成tensor

        return data1,data2, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images1,images2, labels = tuple(zip(*batch))

        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        # images4 = torch.stack(images4,dim=0)
        labels = torch.as_tensor(labels)
        return images1,images2, labels