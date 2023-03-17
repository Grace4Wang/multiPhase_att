import torch.nn as nn
import torch.nn.functional as F
import torch

'''
三模态融合的LeNet做MVI分类
参考文章：Prediction of Microvascular Invasion of Hepatocellular Carcinoma Based on Contrast-Enhanced MR
 and 3D Convolutional Neural Networks
代码参考：https://github.com/Lsx0802/3DCNN_DSN/blob/main/3DCNN_DSN/3D_concat.py
原文DOI:10.3389/fonc.2021.588010
'''

class LeNetFusion(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNetFusion, self).__init__()

        # modal 1 #
        self.conv11 = nn.Conv3d(1, 32, 3, padding='same')
        self.pool11 = nn.MaxPool3d(2, stride=2)

        self.conv12 = nn.Conv3d(32, 64, 3, padding='same')
        self.pool12 = nn.MaxPool3d(2, stride=2)

        self.conv13 = nn.Conv3d(64, 64, 3, padding='same')
        self.pool13 = nn.MaxPool3d(2, stride=2)

        self.fc11 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout11 = nn.Dropout(0.5)
        self.fc12 = nn.Linear(1024, 32)
        self.dropout12 = nn.Dropout(0.5)

        # modal 2 #
        self.conv21 = nn.Conv3d(1, 32, 3, padding='same')
        self.pool21 = nn.MaxPool3d(2, stride=2)

        self.conv22 = nn.Conv3d(32, 64, 3, padding='same')
        self.pool22 = nn.MaxPool3d(2, stride=2)

        self.conv23 = nn.Conv3d(64, 64, 3, padding='same')
        self.pool23 = nn.MaxPool3d(2, stride=2)

        self.fc21 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout21 = nn.Dropout(0.5)
        self.fc22 = nn.Linear(1024, 32)
        self.dropout22 = nn.Dropout(0.5)

        # modal 3 #
        self.conv31 = nn.Conv3d(1, 32, 3, padding='same')
        self.pool31 = nn.MaxPool3d(2, stride=2)

        self.conv32 = nn.Conv3d(32, 64, 3, padding='same')
        self.pool32 = nn.MaxPool3d(2, stride=2)

        self.conv33 = nn.Conv3d(64, 64, 3, padding='same')
        self.pool33 = nn.MaxPool3d(2, stride=2)

        self.fc31 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout31 = nn.Dropout(0.5)
        self.fc32 = nn.Linear(1024, 32)
        self.dropout32 = nn.Dropout(0.5)

        self.fcfusion = nn.Linear(32 * 3, num_classes)

    def forward(self, x1, x2, x3):
        # stage 1
        x1 = F.relu(self.conv11(x1))

        x2 = F.relu(self.conv21(x2))

        x3 = F.relu(self.conv31(x3))

        x1 = self.pool11(x1)
        x2 = self.pool21(x2)
        x3 = self.pool31(x3)

        # stage 2
        x1 = F.relu(self.conv12(x1))

        x2 = F.relu(self.conv22(x2))

        x3 = F.relu(self.conv32(x3))

        x1 = self.pool12(x1)
        x2 = self.pool22(x2)
        x3 = self.pool32(x3)

        # stage 3
        x1 = F.relu(self.conv13(x1))

        x2 = F.relu(self.conv23(x2))

        x3 = F.relu(self.conv33(x3))

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


        #ABAF

        x = self.fcfusion(x)  # output(2)
        return x
