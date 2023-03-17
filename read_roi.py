import SimpleITK
import numpy as np
import os


def read_roi(roi_path,patient):
    '''
    根据mask读ROI区域，
    返回患者名，bbox的x_min, x_max, y_min, y_max, z_min, z_max
    以及宽度：x_max - x_min, 高度：y_max - y_min, 深度：z_max - z_min,
    以及bbox的中间点坐标(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


    :param roi_path:
    :param patient:
    :return:
    '''
    img = SimpleITK.ReadImage(roi_path)
    img_data = SimpleITK.GetArrayFromImage(img)

    values = np.array(img_data)
    a = values.shape

    x_min, x_max, y_min, y_max, z_min, z_max = 10000, 0, 10000, 0, 10000, 0
    tmp_z = []
    for i in range(a[0]):
        slice = values[i]
        if slice.max() > 0:
            tmp_z.append(i)
            idx = np.array(np.where(slice != 0))

            x_min = idx[0, 0]
            x_max = idx[0, -1]
            temp=np.sort(idx[1])
            y_min = temp[0]
            y_max = temp[-1]

    z_min = tmp_z[0]
    z_max = tmp_z[-1]
    return patient,x_min, x_max, y_min, y_max, z_min, z_max, x_max - x_min, y_max - y_min, z_max - z_min, (x_min + x_max) / 2, (
                y_min + y_max) / 2, (z_min + z_max) / 2
    #