import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from Att import NONLocalBlock3D
from CoAtt_4Pch import CoAtt3D
import torch

'''
特征提取的attention融合
参考文章”Attention guided discriminative feature learning and adaptive fusion for grading hepatocellular 
carcinoma with Contrast-enhanced MR“ 
代码参考：https://github.com/Lsx0802/discriminative
原文DOI:10.1016/j.compmedimag.2022.102050
'''

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

        # self.fc11 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.fc11 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout11 = nn.Dropout(0.5)
        # self.fc12 = nn.Linear(1024, 32)
        self.fc12 = nn.Linear(1024, 32)
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

        self.fc21 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout21 = nn.Dropout(0.5)
        self.fc22 = nn.Linear(1024, 32)
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

        self.fc31 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout31 = nn.Dropout(0.5)
        self.fc32 = nn.Linear(1024, 32)
        self.dropout32 = nn.Dropout(0.5)

        # modal 4 #
        self.conv41 = nn.Conv3d(1, 32, 3, padding=1)
        self.pool41 = nn.MaxPool3d(2, stride=2)
        self.nl41 = NONLocalBlock3D(32)

        self.conv42 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool42 = nn.MaxPool3d(2, stride=2)
        self.nl42 = NONLocalBlock3D(64)

        self.conv43 = nn.Conv3d(64, 64, 3, padding=1)
        self.pool43 = nn.MaxPool3d(2, stride=2)
        self.nl43 = NONLocalBlock3D(64)

        self.fc41 = nn.Linear(64 * 2 * 2 * 2, 1024)
        self.dropout41 = nn.Dropout(0.5)
        self.fc42 = nn.Linear(1024, 32)
        self.dropout42 = nn.Dropout(0.5)


        ### fusion
        self.Co1 = CoAtt3D(in_channels=32)
        self.Co2 = CoAtt3D(in_channels=64)
        self.Co3 = CoAtt3D(in_channels=64)
        self.Co4 = CoAtt3D(in_channels=64)

        '''
            
        '''
        # self.fcfusion = nn.Linear(32 * 3, num_classes)
        self.fcfusion = nn.Linear(32 * 4, num_classes)

    def forward(self, x1, x2, x3, x4):
        # stage 1
        x1 = F.relu(self.conv11(x1))
        x1 = self.nl11(x1)

        x2 = F.relu(self.conv21(x2))
        x2 = self.nl21(x2)

        x3 = F.relu(self.conv31(x3))
        x3 = self.nl31(x3)

        x4 = F.relu(self.conv41(x4))
        x4 = self.nl41(x4)

        x1, x2, x3, x4 = self.Co1(x1, x2, x3, x4)

        x1 = self.pool11(x1)
        x2 = self.pool21(x2)
        x3 = self.pool31(x3)
        x4 = self.pool41(x4)

        # stage 2
        x1 = F.relu(self.conv12(x1))
        x1 = self.nl12(x1)

        x2 = F.relu(self.conv22(x2))
        x2 = self.nl22(x2)

        x3 = F.relu(self.conv32(x3))
        x3 = self.nl32(x3)

        x4 = F.relu(self.conv42(x4))
        x4 = self.nl42(x4)

        x1, x2, x3, x4 = self.Co2(x1, x2, x3, x4)

        x1 = self.pool12(x1)
        x2 = self.pool22(x2)
        x3 = self.pool32(x3)
        x4 = self.pool42(x4)

        # stage 3
        x1 = F.relu(self.conv13(x1))
        x1 = self.nl13(x1)

        x2 = F.relu(self.conv23(x2))
        x2 = self.nl23(x2)

        x3 = F.relu(self.conv33(x3))
        x3 = self.nl33(x3)

        x4 = F.relu(self.conv43(x4))
        x4 = self.nl43(x4)

        x1, x2, x3, x4 = self.Co3(x1, x2, x3, x4)

        x1 = self.pool13(x1)
        x2 = self.pool23(x2)
        x3 = self.pool33(x3)
        x4 = self.pool43(x4)

        # stage 4
        x1 = x1.view(-1, 64 * 2 * 2 * 2)
        x1 = F.relu(self.fc11(x1))  # output(1024)
        x1 = self.dropout11(x1)
        x1 = F.relu(self.fc12(x1))  # output(32)
        f1 = x1
        x1 = self.dropout12(x1)

        x2 = x2.view(-1, 64 * 2 * 2 * 2)
        x2 = F.relu(self.fc21(x2))  # output(1024)
        x2 = self.dropout21(x2)
        x2 = F.relu(self.fc22(x2))  # output(32)
        f2 = x2
        x2 = self.dropout22(x2)

        x3 = x3.view(-1, 64 * 2 * 2 * 2)
        x3 = F.relu(self.fc31(x3))  # output(1024)
        x3 = self.dropout31(x3)
        x3 = F.relu(self.fc32(x3))  # output(32)
        f3 = x3
        x3 = self.dropout32(x3)

        x4 = x4.view(-1, 64 * 2 * 2 * 2)
        x4 = F.relu(self.fc41(x4))  # output(1024)
        x4 = self.dropout41(x4)
        x4 = F.relu(self.fc42(x4))  # output(32)
        f4 = x4
        x4 = self.dropout42(x4)

        # stage 5
        x = torch.cat((x1, x2, x3, x4), dim=1)
        t = x
        # plt.imshow(t)
        x = self.fcfusion(x)  # output(2)


        return x, f1, f2, f3, f4, t
