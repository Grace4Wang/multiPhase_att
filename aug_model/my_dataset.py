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

    def __init__(self, data_dir, txt, transform=None):
        super(MyDataSet, self).__init__()

        data = []
        fh = open(txt, 'r')    # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  #按行循环txt文本中的内容
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，即删除换行符
            words = line.split(' ')  # 按空格来切分内容
            data.append((words[0], int(words[1])))  #保存患者名及其标签标签

        self.data_dir = data_dir
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        fn, label = self.data[index]
        image_path = os.path.join(self.data_dir, fn)

        img = SimpleITK.ReadImage(image_path)
        img = SimpleITK.GetArrayFromImage(img)

        img = np.array(img, dtype='float32')
        # img = np.array(img, dtype=object)

        image=img[np.newaxis,:]     #给数据增加一个值为1的channel维度

        '''
        性能不好原因可能出现在这里，
        2D图像做读取时，用Image.open（）或cv2.read（）来读取会读出原图的channel，RGB为3，灰度图为1.
        而我们的数据为3D数据，应该有对应的3D读取方式，我这里用SimpleITK读出来的没有channel维度，故直接增加了一个chennel维度以适应模型
        具体可以参考网上的医学图像3D分割算法，比如3DUnet是如何读取3D的医学图像的，以及是如何输入进网络的。
        '''

        data=torch.from_numpy(image) #转成tensor
        return data, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels