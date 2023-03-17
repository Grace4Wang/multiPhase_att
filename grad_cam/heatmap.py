import os
import numpy as np

from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
import torch.nn.functional as F
import SimpleITK as sitk


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


def main():
    # 这个下面放置你网络的代码，因为载入权重的时候需要读取网络代码，这里我建议直接从自己的训练代码中原封不动的复制过来即可，我这里因为跑代码使用的是Resnet，所以这里将resent的网络复制到这里即可
    class LeNet(nn.Module):
        def __init__(self, num_classes=2):
            super(LeNet, self).__init__()
            # self.conv1 = nn.Conv3d(1, 32, 3,padding='same')
            self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
            self.pool1 = nn.MaxPool3d(2, stride=2)
            # self.conv2 = nn.Conv3d(32, 64, 3,padding='same')
            self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
            self.pool2 = nn.MaxPool3d(2, stride=2)
            # self.conv3 = nn.Conv3d(64, 64, 3,padding='same')
            self.conv3 = nn.Conv3d(64, 64, 3, padding=1)
            self.pool3 = nn.MaxPool3d(2, stride=2)
            # self.fc1 = nn.Linear(64*2*2*2, 1024)
            self.fc1 = nn.Linear(64 * 2 * 2 * 2, 500)
            self.dropout1 = nn.Dropout(0.5)
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

            x = x.view(-1, 64 * 2 * 2 * 2)  # output(64*2*2*2)
            # print(x.shape)
            x = F.relu(self.fc1(x))  # output(1024)
            x = self.dropout1(x)
            # print(x.shape)
            x = F.relu(self.fc2(x))  # output(32)
            x = self.dropout2(x)
            # print(x.shape)
            x = self.fc3(x)  # output(2)
            return x


    net = LeNet()

    device = torch.device("cpu")
    net.load_state_dict(
        torch.load("E:/LSX/Bayer/bbox_ing/aug_model/weights/1_1_AP.pth", map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可

    print(net)
    target_layers = [net.conv2]  # 这里是 看你是想看那一层的输出，我这里是打印的resnet最后一层的输出，你也可以根据需要修改成自己的
    print(target_layers)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 导入图片
    img_path = "E:/Bayer/data/val/AP/003.nii"  # 这里是导入你需要测试图片
    image_size = 16  # 训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    # img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    # image = sitk.GetArrayFromImage(img_sitk)
    # Input_patch_size = [64, image.shape[1], image.shape[2], 1]
    # ct_patch_chunk_List, last_patch_number = Get_image_array_Array_and_give_chunk(image_array=image,
    #                                                                               patch_slice_slice=64)

    img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(img_sitk)

    # img = np.array(img, dtype='float32')

    # img = np.array(img, dtype=object)

    image = img[np.newaxis, :]  # 给数据增加一个值为1的channel维度
    image = torch.from_numpy(image)  # 转成tensor
    print(image.shape)

    # img = Image.open(img_path).convert('RGB')  # 将图片转成RGB格式的
    # img = np.array(img, dtype=np.uint8)  # 转成np格式
    # img = center_crop_img(img, image_size)  # 将测试图像裁剪成跟训练图片尺寸相同大小的

    # [C, H, W]
    # img_tensor = data_transform(img)  # 简单预处理将图片转化为张量
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(image, dim=0)  # 增加一个batch维度
    print(input_tensor.shape)
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    print(grayscale_cam.shape)
    grayscale_cam = grayscale_cam[0, :]

    # visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
    #                                   grayscale_cam,
    #                                   use_rgb=True)

    s_itk_image = sitk.GetImageFromArray(grayscale_cam)
    # print(img_sitk.size)
    s_itk_image.CopyInformation(img_sitk)
    # sitk.WriteImage(s_itk_image, "E:/Bayer/data/figures/AP/003.nii.gz")

    # plt.imshow(visualization)
    # plt.savefig('./result.png')  # 将热力图的结果保存到本地当前文件夹
    # plt.show()


if __name__ == '__main__':
    main()