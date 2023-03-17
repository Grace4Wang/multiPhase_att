# coding: utf-8
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk

# 类的作用
# 1.编写梯度获取hook
# 2.网络层上注册hook
# 3.运行网络forward backward
# 4.根据梯度和特征输出热力图

class ShowGradCam:
    def __init__(self,conv_layer):
        assert isinstance(conv_layer,torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.full_backward_hook)
        self.grad_res = []
        self.feature_res = []

    def full_backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self,module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[0:4], dtype=np.float32)  # cam shape (H, W)
        print(feature_map.shape)
        weights = np.mean(grads, axis=(0, 1, 2))  #
        # print(weights.shape)

        for i, w in enumerate(weights):
            # cam += w * feature_map[:, :, :, i]
            cam += w * feature_map[:,:,i]

        cam = np.maximum(cam, 0)
        cam = resize(cam, (16, 16,16))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self,input_img,i):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        fmap, grads_val = [],[]
        if isinstance(input_img,str):
            img_sitk = sitk.ReadImage(input_img, sitk.sitkFloat32)
            image = sitk.GetArrayFromImage(img_sitk)
        img_size = [16,input_img.shape[1], input_img.shape[2], i]
        print(img_size)
        feares = self.feature_res[0]
        feares = torch.stack(feares,dim=0)
        # print(type(feares))
        fmap = feares.cpu().data.numpy().squeeze()
        # fmap = np.array(feares)
        # graval = self.grad_res[0]
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        # grads_val = np.array(self.grad_res[0])
        cam = self.gen_cam(fmap, grads_val)
        cam = resize(cam, img_size)
        # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)/255.
        # cam = heatmap + np.float32(input_img/255.)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # cam = cam / np.max(cam)*255
        return cam
