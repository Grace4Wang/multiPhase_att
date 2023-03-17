import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def imshow(img):
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    image = r'H:/Bayer/data/figures/AP/006.nii.gz'
    print(read_img(image).shape)

    # t1 = (read_img(image)[5, :, :]).astype(np.uint8)
    t2 = (read_img(image)[10, :, :]).astype(np.uint8)

    # imshow(t1)
    imshow(t2)
