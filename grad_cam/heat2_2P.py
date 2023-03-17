# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import argparse
import sys
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from Att import NONLocalBlock3D
from CoAtt_2P import CoAtt3D
import torchvision.transforms as transforms
from utils_Att import ShowGradCam
import SimpleITK as sitk
from skimage.transform import resize
import nibabel as nib
from Guided_GradCAM_3D_config import*
from my_dataset_fusion import MyDataSet


class LeNetAtt(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNetAtt, self).__init__()

        # modal 1 #
        self.conv11 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool11 = nn.MaxPool3d(2, stride=2)
        self.nl11 = NONLocalBlock3D(32)

        self.conv12 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool12 = nn.MaxPool3d(2, stride=2)
        self.nl12 = NONLocalBlock3D(64)

        self.conv13 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool13 = nn.MaxPool3d(2, stride=2)
        self.nl13 = NONLocalBlock3D(64)

        self.fc11 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout11 = nn.Dropout(0.5)
        self.fc12 = nn.Linear(500, 50)
        self.dropout12 = nn.Dropout(0.5)

        # modal 2 #
        self.conv21 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool21 = nn.MaxPool3d(2, stride=2)
        self.nl21 = NONLocalBlock3D(32)

        self.conv22 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool22 = nn.MaxPool3d(2, stride=2)
        self.nl22 = NONLocalBlock3D(64)

        self.conv23 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool23 = nn.MaxPool3d(2, stride=2)
        self.nl23 = NONLocalBlock3D(64)

        self.fc21 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout21 = nn.Dropout(0.5)
        self.fc22 = nn.Linear(500, 50)
        self.dropout22 = nn.Dropout(0.5)


        ### fusion
        self.Co1 = CoAtt3D(in_channels=32)
        self.Co2 = CoAtt3D(in_channels=64)
        self.Co3 = CoAtt3D(in_channels=64)

        self.fcfusion = nn.Linear(50 * 2, num_classes)

    def forward(self, x1, x2):
        # stage 1
        x1 = F.relu(self.conv11(x1))
        x1 = self.nl11(x1)

        x2 = F.relu(self.conv21(x2))
        x2 = self.nl21(x2)


        x1, x2 = self.Co1(x1, x2)

        x1 = self.pool11(x1)
        x2 = self.pool21(x2)


        # stage 2
        x1 = F.relu(self.conv12(x1))
        x1 = self.nl12(x1)

        x2 = F.relu(self.conv22(x2))
        x2 = self.nl22(x2)


        x1, x2 = self.Co2(x1, x2)

        x1 = self.pool12(x1)
        x2 = self.pool22(x2)

        # stage 3
        x1 = F.relu(self.conv13(x1))
        x1 = self.nl13(x1)

        x2 = F.relu(self.conv23(x2))
        x2 = self.nl23(x2)

        x1, x2 = self.Co3(x1, x2)

        x1 = self.pool13(x1)
        x2 = self.pool23(x2)

        # stage 4
        x1 = x1.view(-1, 64 * 2 * 2 * 2)
        x1 = F.relu(self.fc11(x1))  # output(1024)
        x1 = self.dropout11(x1)
        x1 = F.relu(self.fc12(x1))  # output(32)
        x1 = self.dropout12(x1)

        x2 = x2.view(-1, 64 * 2 * 2 * 2)
        x2 = F.relu(self.fc21(x2))  # output(1024)
        x2 = self.dropout21(x2)
        x2 = F.relu(self.fc22(x2))  # output(32)
        x2 = self.dropout22(x2)


        # stage 5
        x = torch.cat((x1, x2), dim=1)

        # ABAF
        # # w_fusion_11 = weight_variable([150, 50])
        # w_fusion_11 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # # b_fusion_11 = bias_variable([50])
        # b_fusion_11 = torch.zeros([50])+0.1
        # w_fusion_21 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # b_fusion_21 = torch.zeros([50])+0.1
        # w_fusion_31 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # b_fusion_31 = torch.zeros([50])+0.1
        #
        # w_fusion_12 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_12 = torch.zeros([50]) + 0.1
        # w_fusion_22 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_22 = torch.zeros([50]) + 0.1
        # w_fusion_32 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_32 = torch.zeros([50]) + 0.1
        #
        # f11 = torch.matmul(concat1, w_fusion_11) + b_fusion_11
        # f21 = torch.matmul(concat1, w_fusion_21) + b_fusion_21
        # f31 = torch.matmul(concat1, w_fusion_31) + b_fusion_31
        #
        # alpha1 = torch.sigmoid(f11)
        # alpha2 = torch.sigmoid(f21)
        # alpha3 = torch.sigmoid(f31)
        #
        # f1 = torch.multiply(alpha1, fc_out_12)
        # f2 = torch.multiply(alpha2, fc_out_22)
        # f3 = torch.multiply(alpha3, fc_out_32)
        #
        # concat2 = torch.cat((x1, x2, x3), dim=1)
        # f12 = torch.matmul(concat2, w_fusion_12) + b_fusion_12
        # f22 = torch.matmul(concat2, w_fusion_22) + b_fusion_22
        # f32 = torch.matmul(concat2, w_fusion_32) + b_fusion_32
        #
        # f14 = torch.reduce_sum(torch.sigmoid(f12))
        # f24 = torch.reduce_sum(torch.sigmoid(f22))
        # f34 = torch.reduce_sum(torch.sigmoid(f32))
        #
        # f4 = torch.stack([f14, f24, f34])
        # beta = torch.nn.softmax(f4)
        #
        # f13 = beta[0]
        # f23 = beta[1]
        # f33 = beta[2]

        x = self.fcfusion(x)  # output(2)
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
    # name = args.name
    # 保存路径
    savefile = 'E:/Bayer/data/heatmap/2P/PP'
    # 验证集
    filepath1 = 'E:/Bayer/data/val/AP'
    filepath2 = 'E:/Bayer/data/val/PP'

    # 叠加
    img_valpath = 'E:/Bayer/data/trans/PP'
    filenames = os.listdir(filepath1)  # 读取nii文件夹
    for f in filenames:
        # 开始读取nii文件
        # path_img = os.path.join(filepath1, f)
        # img = nib.load(path_img) # 读取nii
        # img_fdata = img.get_fdata()
        fname = f.replace('.nii', '') # 去掉nii的后缀名
        valname = f.replace('.nii','.png')
        img_f_path = os.path.join(savefile, fname)
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path) # 新建文件夹
        valpath = os.path.join(img_valpath,fname)


    # path_img1 = os.path.join("E:/Bayer/data/val/AP/003.nii")
    # path_img2 = os.path.join("E:/Bayer/data/val/HBP/003.nii")
    # path_img3 = os.path.join("E:/Bayer/data/val/PP/003.nii")
    # load image
        data_path_val1 = "E:/Bayer/data/val/AP/"   # 预测数据的路径
        data_path_val2 = "E:/Bayer/data/val/PP/"

        path_net = os.path.join("E:/LSX/Bayer/bbox_ing/aug_model/weights/1_1_AP_PP.pth") # in this example not use

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        classes = ('0', '1')

        # 图片读取；网络加载
        image_path1 = os.path.join(data_path_val1+f)
        img_sitk1 = sitk.ReadImage(image_path1, sitk.sitkFloat32)  # H*W*C
        img1 = sitk.GetArrayFromImage(img_sitk1)
        image1 = img1[np.newaxis, :]  # 给数据增加一个值为1的channel维度
        image1 = torch.from_numpy(image1)  # 转成tensor
        input_tensor1 = torch.unsqueeze(image1, dim=0)  # 增加一个batch维度

        image_path2 = os.path.join(data_path_val2+f)
        img_sitk2 = sitk.ReadImage(image_path2, sitk.sitkFloat32)  # H*W*C
        img2 = sitk.GetArrayFromImage(img_sitk2)
        image2 = img2[np.newaxis, :]  # 给数据增加一个值为1的channel维度
        image2 = torch.from_numpy(image2)  # 转成tensor
        input_tensor2 = torch.unsqueeze(image2, dim=0)  # 增加一个batch维度

        #
        #
        # Input_patch_size_slice_number = INPUT_PATCH_SIZE_SLICE_NUMBER
        # Input_patch_size = [Input_patch_size_slice_number, img1.shape[1], img1.shape[2], 1]
        # ct_patch_chunk_List,last_patch_number = Get_image_array_Array_and_give_chunk(image_array=img1,patch_slice_slice=Input_patch_size_slice_number)

        device = torch.device("cpu")
        net = LeNetAtt(num_classes=2).to(device)
        net.load_state_dict(
            torch.load(path_net, map_location=device),strict=False)  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
        net.eval()

        gradCam = ShowGradCam(net.Co2) #............................. def which layer to show

        # forward
        output = net(input_tensor1.to(device),input_tensor2.to(device))
        idx = np.argmax(output.cpu().data.numpy())
        print("predict: {}".format(classes[idx]))

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward(retain_graph=True)

        # save result
        for i in range(1, 17):
            impath =  os.path.join(valpath,str(i)+'.png')
            img_val = cv2.imread(impath, 1)  # H*W*C
            cam = gradCam.show_on_img(img_val,i) #.......................... show gradcam on target pic
            cv2.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), cam)
    # print('save gradcam result in grad_feature.jpg')
    # s_itk_image = sitk.GetImageFromArray(gradCam)
    # s_itk_image.CopyInformation(img_sitk)
    # sitk.WriteImage(s_itk_image, "E:/Bayer/data/figures/AP/try/003.nii.gz")
