import os
import SimpleITK
import numpy as np
import torch
from show_img import show_3d_np_array

#
def read_dicom_series(path_dcm):
    '''
    按序列读取dicom文件
    以numpy.array格式返回患者影像value

    :param path_dcm:
    :return:
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_dcm)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_data = SimpleITK.GetArrayFromImage(image)
    # values = np.array(img_data)
    values = torch.from_numpy(img_data.astype(float))
    return values


def extract_tumor_bbox(location_txt_name, values, save_extract_path, save_val_path, patient_id):
    '''
    根据location_txt来获取患者肿瘤区域，区域边缘各增加两个像素


    :param save_val_path:
    :param location_txt_name:
    :param values:
    :param save_extract_path:
    :param patient_id:
    :return:
    '''
    with open(str(location_txt_name) + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[0:-1]
            a = line.split(' ')
            b = a[0].split('.')
            if b[0] == patient_id:

                value_z, value_x, value_y = values.shape

                bbox_x_min,  bbox_x_max, bbox_y_min,bbox_y_max,  bbox_z_min,bbox_z_max = \
                    max(float(a[1])-2,0),min(float(a[2])+2,value_x),max(float(a[3])-2,0),min(float(a[4])+2,value_y),\
                    max(float(a[5])-2,0),min(float(a[6])+2,value_z)

                extract_values = values[int(bbox_z_min):int(bbox_z_max), int(bbox_x_min):int(bbox_x_max),
                                 int(bbox_y_min):int(bbox_y_max)] #H,W,D

                save_path = os.path.join(save_extract_path, a[0])
                save_val = os.path.join(save_val_path, a[0])

                # extract_values=np.resize(extract_values,(32,32,32))
                # ori = torch.from_numpy(extract_values.astype(float))


                # extract_values = torch.from_numpy(extract_values.astype(float))
                extract_values = extract_values[None, None, :, :, :]

                ori = extract_values
                ori = torch.nn.functional.interpolate(ori, size=(16,16,16),
                                                             mode='trilinear')

                extract_values = torch.nn.functional.interpolate(extract_values, size=(32,32,32),
                                                             mode='trilinear')
                '''
                extract_values = torch.from_numpy(extract_values.astype(float))
                extract_values = extract_values[None, None, :, :, :]
                extract_values = torch.nn.functional.interpolate(extract_values, size=(32,32,32),
                                                             mode='trilinear')
                可以使用pytorch中的torch.nn.functional.interpolate ()实现插值和上采样。 
                
                在这里输出验证的图 ori文件夹
                '''
                outputVal = SimpleITK.GetImageFromArray(ori)
                SimpleITK.WriteImage(outputVal,save_val)

                out = SimpleITK.GetImageFromArray(extract_values)
                SimpleITK.WriteImage(out, save_path)
                # show_3d_np_array(extract_values)
