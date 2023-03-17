# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils_2 import ShowGradCam
import SimpleITK as sitk
from skimage.transform import resize
from Guided_GradCAM_3D_config import*
import imageio
import nibabel as nib # nii格式一般都会用到这个包


class LeNet(nn.Module):
    def __init__(self,num_classes=2):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv3d(1, 32, 3,padding='same')
        self.conv1 = nn.Conv3d(1, 32, 3,padding=1)
        self.pool1 = nn.MaxPool3d(2,stride=2)
        # self.conv2 = nn.Conv3d(32, 64, 3,padding='same')
        self.conv2 = nn.Conv3d(32, 64, 3,padding=1)
        self.pool2 = nn.MaxPool3d(2,stride=2)
        # self.conv3 = nn.Conv3d(64, 64, 3,padding='same')
        self.conv3 = nn.Conv3d(64, 64, 3,padding=1)
        self.pool3 = nn.MaxPool3d(2,stride=2)
        # self.fc1 = nn.Linear(64*2*2*2, 1024)
        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout1=nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(500, 50)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(32, num_classes)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))  # input(1, 16,16,16) output(16, 16,16,16)
        # print(x.shape)
        x = self.pool1(x)  # output(16, 8,8,8)
        # print(x.shape)

        # x=DotProduct(x)

        x = F.relu(self.conv2(x))  # output(32, 8,8,8)
        # print(x.shape)
        x = self.pool2(x)  # output(32, 4,4,4)
        # print(x.shape)

        # x = DotProduct(x)

        x = F.relu(self.conv3(x))  # output(64, 4,4,4)
        # print(x.shape)
        x = self.pool3(x)  # output(64, 2,2,2)
        # print(x.shape)

        # x = DotProduct(x)

        x = x.view(-1, 64*2*2*2)  # output(64*2*2*2)
        # print(x.shape)
        x = F.relu(self.fc1(x))  # output(1024)
        x = self.dropout1(x)
        # print(x.shape)
        x = F.relu(self.fc2(x))  # output(32)
        x = self.dropout2(x)
        # print(x.shape)
        x = self.fc3(x)  # output(2)
        return x

def Get_image_array_Array_and_give_chunk(image_array,patch_slice_slice):

    Devide_integer=image_array.shape[0] // patch_slice_slice
    Reminder= image_array.shape[0] % patch_slice_slice
    print('CT Volume_Shape={}'.format(image_array.shape))
    print('Devide_integer={}'.format(Devide_integer))
    print('Reminder={}'.format(Reminder))
    print('Total of {} + {} ={} Should ={}'.format(patch_slice_slice*Devide_integer,Reminder,patch_slice_slice*Devide_integer+Reminder,image_array.shape[0]))

    lastpatch_starts_from= (image_array.shape[0])-patch_slice_slice
    print(lastpatch_starts_from)

    patch_list=[]
    patch_start=0
    patch_end=patch_slice_slice
    for i in range(Devide_integer):
        #print(patch_start)
        #print(patch_end)
        ct_volume=image_array[patch_start:patch_end,:,:]
        #print(ct_volume.shape)
        patch_list.append(ct_volume)
        patch_start+=patch_slice_slice
        patch_end+=patch_slice_slice

    last_slice_number_would_be=image_array.shape[0]
    print(last_slice_number_would_be)
    last_patch_When_making_nifty=(patch_slice_slice)-Reminder
    print(last_patch_When_making_nifty)
    Slice_will_start_from_here=last_slice_number_would_be-patch_slice_slice
    print(Slice_will_start_from_here)
    last_patch=image_array[Slice_will_start_from_here:,:,:]
    last_patch.shape
    patch_list.append(last_patch)

    return patch_list,last_patch_When_making_nifty

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = resize(img,(16, 16))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input

def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


if __name__ == '__main__':
    savefile = 'E:/Bayer/data/heatmap/T1WI'
    img_valpath = 'E:/Bayer/data/trans/T1WI'
    filepath = 'E:/Bayer/data/val/T1WI'
    predict = []
    data1 = []
    filenames = os.listdir(filepath)  # 读取nii文件夹
    for f in filenames:
        # 开始读取nii文件
        path_img = os.path.join(filepath, f)
        img = nib.load(path_img) # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '') # 去掉nii的后缀名
        valname = f.replace('.nii','.png')
        img_f_path = os.path.join(savefile, fname)
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path) # 新建文件夹
        valpath  = os.path.join(img_valpath,fname)
        # if not os.path.exists(img_f_path):
        #     os.mkdir(img_f_path) # 新建文件夹

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # path_img = os.path.join("E:/Bayer/data/val/AP/003.nii")
        path_net = os.path.join("E:/LSX/Bayer/bbox_ing/aug_model/weights/1_1_T1WI.pth") # in this example not use

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        classes = ('0', '1')

        # 图片读取；网络加载
        img_sitk = sitk.ReadImage(path_img, sitk.sitkFloat32)  # H*W*C
        img = sitk.GetArrayFromImage(img_sitk)
        image = img[np.newaxis, :]  # 给数据增加一个值为1的channel维度
        image = torch.from_numpy(image)  # 转成tensor
        input_tensor = torch.unsqueeze(image, dim=0)  # 增加一个batch维度
    # print(input_tensor.shape)

        Input_patch_size_slice_number = INPUT_PATCH_SIZE_SLICE_NUMBER
        Input_patch_size = [Input_patch_size_slice_number, img.shape[1], img.shape[2], 1]
        ct_patch_chunk_List,last_patch_number = Get_image_array_Array_and_give_chunk(image_array=img,patch_slice_slice=Input_patch_size_slice_number)
        net = LeNet()
        device = torch.device("cpu")
        net.load_state_dict(
            torch.load(path_net, map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
        net.eval()

        gradCam = ShowGradCam(net.conv1) #............................. def which layer to show

        # forward
        output = net(input_tensor)
        data1.extend(output.detach().numpy())
        # np.savetxt("test_T1WI.csv", data1)
        idx = np.argmax(output.cpu().data.numpy())
        predict.extend(classes[idx])
        # data2 = predict.numpy()
        # np.savetxt("predict_T1WI.csv", predict, fmt='%s')
        print("predict: {}".format(classes[idx]))

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward(retain_graph=True)

        # save result
        # for i in range(0, 16):
        #     impath =  os.path.join(valpath,str(i)+'.png')
        #     img_val = cv2.imread(impath, 1)  # H*W*C
        #     cam = gradCam.show_on_img(img_val) #.......................... show gradcam on target pic
        #     cv2.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), cam)

    # print('save gradcam result in grad_feature.jpg')
        # s_itk_image = sitk.GetImageFromArray(gradCam)
    # s_itk_image.CopyInformation(img_sitk)
    # sitk.WriteImage(s_itk_image, "E:/Bayer/data/figures/AP/try/003.nii.gz")