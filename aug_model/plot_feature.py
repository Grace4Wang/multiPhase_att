# 查看和显示nii.gz文件

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

example_filename = 'H:/Bayer/data/figures/AP/006.nii.gz'
# example_filename = 'E:/Bayer/data/figures/AP_try/pool3/attention_map_1_0_0.nii.gz'
img = nib.load(example_filename)
print(img)
print(img.header['db_name'])  # 输出头信息

# shape有四个参数 patient001_4d.nii.gz
# shape有三个参数 patient001_frame01.nii.gz   patient001_frame12.nii.gz
# shape有三个参数  patient001_frame01_gt.nii.gz   patient001_frame12_gt.nii.gz
width, height, queue = img.dataobj.shape
# OrthoSlicer3D(img.dataobj).show()
print(width)
print(height)
print(queue)


num = 1
for i in range(0, width, 1):
    img_arr = img.dataobj[i, :, :]
    # plt.subplot(5, 4, num)
    # plt.imshow(img_arr, cmap='gray')
    plt.imshow(img_arr)
    num += 1

plt.show()
