import numpy as np
import SimpleITK
import os

import torch
from tqdm import tqdm


'''
滑块增强，16*16*16
验证集直接reszie成16*16*16
'''


save_path = r'E:\Bayer\data\augmentation'
path_1 = r'E:\Bayer\data\extract_roi_shape'
path_2 = tqdm(os.listdir(path_1))
for p1 in path_2:
    path_3 = os.path.join(path_1, p1)
    path_4 = tqdm(os.listdir(path_3))
    for p2 in path_4:
        path_5 = os.path.join(path_3, p2)
        data = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path_5))
        value_z, value_x, value_y = data.shape

        save_path_2 = os.path.join(save_path, p1)
        if os.path.exists(save_path_2) == False:
            os.makedirs(save_path_2)

        i=0
        while i < (int(value_z / 2)):
            j=0
            while j < (int(value_x / 2)):
                k=0
                while k < (int(value_y / 2)):
                    value = data[i:i + 18, j:j + 18, k:k + 18]
                    save_path_3 = os.path.join(save_path_2,
                                               p2[0:3] + '_' + str(i) + '_' + str(j) + '_' + str(k) + p2[3:])

                    out = SimpleITK.GetImageFromArray(value)
                    SimpleITK.WriteImage(out, save_path_3)

                    k = k + 2
                j = j + 2
            i = i + 2

        # ori = np.resize(data, (16, 16, 16))
        # ori = torch.from_numpy(data.astype(float))
        # ori = ori[None, None, :, :, :]
        # ori = torch.nn.functional.interpolate(ori, size=(16, 16, 16),
        #                                       mode='trilinear')
        # ori去掉 改到
        # save_path_4 = os.path.join(save_path, 'ori', p1)
        # if os.path.exists(save_path_4) == False:
        #     os.makedirs(save_path_4)
        # save_path_5 = os.path.join(save_path_4, p2[0:3] + '_ori' + p2[3:])
        # out = SimpleITK.GetImageFromArray(ori)
        # SimpleITK.WriteImage(out, save_path_5)
        # print('0')
