import os
import SimpleITK
'''
这是一个小demo来对需要删减某些层的mask.nii文件来进行操作
'''

a='446.nii'
path=r'E:\LSX\Bayer\SYS1_PV\nii_3/'+a
save_path=r'E:\LSX\Bayer\SYS1_PV\nii_3_3'
if os.path.exists(save_path)==False:
    os.makedirs(save_path)

img = SimpleITK.ReadImage(path)
img_data = SimpleITK.GetArrayFromImage(img)
img=img_data[1:-1]
out=SimpleITK.GetImageFromArray(img)
SimpleITK.WriteImage(out,os.path.join(save_path,a))