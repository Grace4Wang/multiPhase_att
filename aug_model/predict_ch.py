import os
import json
import SimpleITK as sitk
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Att import NONLocalBlock3D
from CoAtt import CoAtt3D
from my_dataset_fusion import MyDataSet
# from LeNetAtt import LeNetAtt as create_model

class LeNetAtt(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNetAtt, self).__init__()

        # modal 1 #
        self.conv11 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool11 = nn.MaxPool3d(2, stride=2)
        self.nl11 = NONLocalBlock3D(32)

        self.conv12 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool12 = nn.MaxPool3d(2, stride=2)
        self.nl12 = NONLocalBlock3D(64)

        self.conv13 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool13 = nn.MaxPool3d(2, stride=2)
        self.nl13 = NONLocalBlock3D(64)

        self.fc11 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout11 = nn.Dropout(0.5)
        self.fc12 = nn.Linear(500, 50)
        self.dropout12 = nn.Dropout(0.5)

        # modal 2 #
        self.conv21 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool21 = nn.MaxPool3d(2, stride=2)
        self.nl21 = NONLocalBlock3D(32)

        self.conv22 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool22 = nn.MaxPool3d(2, stride=2)
        self.nl22 = NONLocalBlock3D(64)

        self.conv23 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool23 = nn.MaxPool3d(2, stride=2)
        self.nl23 = NONLocalBlock3D(64)

        self.fc21 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout21 = nn.Dropout(0.5)
        self.fc22 = nn.Linear(500, 50)
        self.dropout22 = nn.Dropout(0.5)

        # modal 3 #
        self.conv31 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool31 = nn.MaxPool3d(2, stride=2)
        self.nl31 = NONLocalBlock3D(32)

        self.conv32 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool32 = nn.MaxPool3d(2, stride=2)
        self.nl32 = NONLocalBlock3D(64)

        self.conv33 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool33 = nn.MaxPool3d(2, stride=2)
        self.nl33 = NONLocalBlock3D(64)

        self.fc31 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout31 = nn.Dropout(0.5)
        self.fc32 = nn.Linear(500, 50)
        self.dropout32 = nn.Dropout(0.5)

        ### fusion
        self.Co1 = CoAtt3D(in_channels=32)
        self.Co2 = CoAtt3D(in_channels=64)
        self.Co3 = CoAtt3D(in_channels=64)

        self.fcfusion = nn.Linear(50 * 3, num_classes)

    def forward(self, x1, x2, x3):
        # stage 1
        x1 = F.relu(self.conv11(x1))
        x1 = self.nl11(x1)

        x2 = F.relu(self.conv21(x2))
        x2 = self.nl21(x2)

        x3 = F.relu(self.conv31(x3))
        x3 = self.nl31(x3)

        x1, x2, x3 = self.Co1(x1, x2, x3)

        x1 = self.pool11(x1)
        x2 = self.pool21(x2)
        x3 = self.pool31(x3)

        # stage 2
        x1 = F.relu(self.conv12(x1))
        x1 = self.nl12(x1)

        x2 = F.relu(self.conv22(x2))
        x2 = self.nl22(x2)

        x3 = F.relu(self.conv32(x3))
        x3 = self.nl32(x3)

        x1, x2, x3 = self.Co2(x1, x2, x3)

        x1 = self.pool12(x1)
        x2 = self.pool22(x2)
        x3 = self.pool32(x3)

        # stage 3
        x1 = F.relu(self.conv13(x1))
        x1 = self.nl13(x1)

        x2 = F.relu(self.conv23(x2))
        x2 = self.nl23(x2)

        x3 = F.relu(self.conv33(x3))
        x3 = self.nl33(x3)

        x1, x2, x3 = self.Co3(x1, x2, x3)

        x1 = self.pool13(x1)
        x2 = self.pool23(x2)
        x3 = self.pool33(x3)

        # stage 4
        x1 = x1.view(-1, 64 * 2 * 2 * 2)
        x1 = F.relu(self.fc11(x1))  # output(1024)
        x1 = self.dropout11(x1)
        x1 = F.relu(self.fc12(x1))  # output(32)
        x1 = self.dropout12(x1)

        x2 = x2.view(-1, 64 * 2 * 2 * 2)
        x2 = F.relu(self.fc21(x2))  # output(1024)
        x2 = self.dropout21(x2)
        x2 = F.relu(self.fc22(x2))  # output(32)
        x2 = self.dropout22(x2)

        x3 = x3.view(-1, 64 * 2 * 2 * 2)
        x3 = F.relu(self.fc31(x3))  # output(1024)
        x3 = self.dropout31(x3)
        x3 = F.relu(self.fc32(x3))  # output(32)
        x3 = self.dropout32(x3)

        # stage 5
        x = torch.cat((x1, x2, x3), dim=1)

        # ABAF

        x = self.fcfusion(x)  # output(2)
        return x

def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2
    pred = []
    data1 = []

    # load image
    data_path_val1 = r"E:\Bayer\data\val/AP"  # 预测数据的路径
    data_path_val2 = r"E:\Bayer\data\val/HBP"
    data_path_val3 = r"E:\Bayer\data\val/PP"
    val_txt_path = r'E:\LSX\Bayer\bbox_ing\aug_model\txt\label.txt'

    val_dataset = MyDataSet(data_path_val1, data_path_val2, data_path_val3, val_txt_path)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=210,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)
    # create model
    model = LeNetAtt(num_classes=num_classes).to(device)

    for step, dataf in enumerate(val_loader):
        images1, images2, images3, labels = dataf
        # pred = model(images1.to(device), images2.to(device), images3.to(device))

    # img_path = ""   #预测数据的路径
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    #
    # img = sitk.ReadImage(img_path)
    # img = sitk.GetArrayFromImage(img)
    # img = img[np.newaxis, :]
    #
    # # [C, H, W，D]
    # img = torch.from_numpy(np.array(img,dtype='float32'))
    # # expand batch_size dimension  # [N,C, H, W，D]
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        # with open(json_path, "r") as f:
        #     class_indict = json.load(f)
        classes = ('0', '1')

        # load model weights
        model_weight_path = "E:/LSX/Bayer/bbox_ing/aug_model/weights/1_1_fusion_3P.pth" #权重路径
        model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)   #加载权重
        model.eval()    #设置为验证模式

        # forward
        output = model(images1.to(device), images2.to(device), images3.to(device))
        data1.extend(output.detach().numpy())
        # np.savetxt("test_T1WI.csv", data1)
        pre_class = torch.max(output,dim=1)[1]
        print(pre_class)
        # idx = np.argmax(output.cpu().data.numpy())
        # pred.extend(classes[idx])
        # data2 = predict.numpy()
        # np.savetxt("predict_T1WI.csv", predict, fmt='%s')
        # print("predict: {}".format(classes[idx]))

        # backward
        model.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward(retain_graph=True)

        # with torch.no_grad():
        #     # predict class
        #     # output = torch.squeeze(model(images1.to(device), images2.to(device), images3.to(device))).cpu()
        #     output = model(images1.to(device), images2.to(device), images3.to(device))
        #     data1.extend(output.detach().numpy())
        #     np.savetxt("test_3P.csv", data1)
        #     # idx = np.argmax(output.cpu().data.numpy())
        #     predict = torch.softmax(output, dim=0)  #类别的概率
        #     predict_cla = torch.argmax(predict).numpy() #选择大概率的类别作为预测类别
        #     # print(idx)
        #     # pred.extend(predict_cla)
        #     # pred.extend(classes[idx])
        #     # print("predict: {}".format(classes[idx]))
        #     print(predict)
        #     print(predict_cla)
            # print(pred)
            # np.savetxt("predict_3P.csv", pred, fmt='%s')

            # backward
            # model.zero_grad()
            # class_loss = comp_class_vec(output)
            # class_loss.backward(retain_graph=True)

            # print("class: {}   prob: {:.3}".format(classes[predict_cla],
            #                                       predict[predict_cla].numpy()))

# if __name__ == '__main__':
#     main()
