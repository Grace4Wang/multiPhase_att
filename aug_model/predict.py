import os
import json
import SimpleITK as sitk
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from LeNet5 import LeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2

    # load image
    img_path = ""  # 预测数据的路径
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    img = img[np.newaxis, :]

    # [C, H, W，D]
    img = torch.from_numpy(np.array(img,dtype='float32'))
    # expand batch_size dimension  # [N,C, H, W，D]
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = LeNet(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "" #权重路径
    model.load_state_dict(torch.load(model_weight_path, map_location=device))   #加载权重
    model.eval()    #设置为验证模式
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)  #类别的概率
        predict_cla = torch.argmax(predict).numpy() #选择大概率的类别作为预测类别

    print("class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy()))

if __name__ == '__main__':
    main()
