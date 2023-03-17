import numpy as np
import SimpleITK
import pyvista as pv
import matplotlib.pyplot as plt


# path=r'E:\Bayer\EOB\bad\1 TU WEI XIN\CE2.mha/CE2.mha'

'''
可视化一个矩阵2D或3D
'''

def show_img(path):
    img = SimpleITK.ReadImage(path) #读取文件内容
    img_data = SimpleITK.GetArrayFromImage(img)

    values=np.array(img_data)
    print(values.shape)

    # 首先类似于新建一个空的矩阵
    grid = pv.UniformGrid()

    # 然后设置维度，如果想给cell填充数据则维度设置为 矩阵shape + 1，
    # 如果是给point填充数据指定为矩阵shape
    # 填充cell，颜色就不是渐变的，是每个块一个颜色，正好和ct的数据一样。
    grid.dimensions = np.array(values.shape)+1

    # 设置（网格）矩阵的值
    grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!

    # plot
    grid.plot(show_edges=False)

def show_3d_np_array(values):
    grid = pv.UniformGrid()
    # 然后设置维度，如果想给cell填充数据则维度设置为 矩阵shape + 1，
    # 如果是给point填充数据指定为矩阵shape
    # 填充cell，颜色就不是渐变的，是每个块一个颜色，正好和ct的数据一样。
    grid.dimensions = np.array(values.shape) + 1

    # 设置（网格）矩阵的值
    grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!

    # plot
    grid.plot(show_edges=False)

def show_slice_array(slice):
    plt.imshow(slice,cmap='gray')
    plt.show()
    plt.pause(1)