import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from Att import NONLocalBlock3D
from CoAtt import CoAtt3D
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

        self.fc11 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout11 = nn.Dropout(0.5)
        self.fc12 = nn.Linear(500, 50)
        # self.dropout12 = nn.Dropout(0.5)
        self.fc13 = nn.Linear(50, 2)

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
        # self.dropout22 = nn.Dropout(0.5)
        self.fc23 = nn.Linear(50, 2)

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
        # self.dropout32 = nn.Dropout(0.5)
        self.fc33 = nn.Linear(50, 2)

        ### fusion
        self.f1 = nn.Linear(150, 50)
        self.f2 = nn.Linear(50, num_classes)

        self.fusion11 = nn.Linear(150, 50)
        self.fusion21 = nn.Linear(150, 50)
        self.fusion31 = nn.Linear(150, 50)

        #
        # self.Co1 = CoAtt3D(in_channels=32)
        # self.Co2 = CoAtt3D(in_channels=64)
        # self.Co3 = CoAtt3D(in_channels=64)
        #
        # self.fcfusion = nn.Linear(32 * 3, num_classes)

    def forward(self, x1, x2, x3, label):
        criterion = nn.CrossEntropyLoss()
        ############# Axial Feature ##################
        conv_out_11 = F.relu(self.nl11(self.conv11(x1)))
        pool_out_11 = self.pool11(conv_out_11)
        conv_out_12 = F.relu(self.nl12(self.conv12(pool_out_11)))
        pool_out_12 = self.pool12(conv_out_12)
        pool_out_12 = pool_out_12.view(-1, 64 * 2 * 2 * 2)
        fc_out_11 = F.relu(self.fc11(pool_out_12))
        dropout_11 = self.dropout11(fc_out_11)
        fc_out_12 = F.relu(self.fc12(dropout_11))

        mid1 = torch.softmax(self.fc13(fc_out_12), dim=1)
        L1 = torch.mean(criterion(mid1, label))

        ################## Coronal Feature #############

        conv_out_21 = F.relu(self.nl21(self.conv21(x2)))
        pool_out_21 = self.pool21(conv_out_21)
        conv_out_22 = F.relu(self.nl22(self.conv22(pool_out_21)))
        pool_out_22 = self.pool22(conv_out_22)
        pool_out_22 = pool_out_22.view(-1, 64 * 2 * 2 * 2)
        fc_out_21 = F.relu(self.fc21(pool_out_22))
        dropout_21 = self.dropout21(fc_out_21)
        fc_out_22 = F.relu(self.fc22(dropout_21))

        mid2 = torch.softmax(self.fc23(fc_out_22), dim=1)
        L2 = torch.mean(criterion(mid2, label))

        ############################## Sagittal ######################

        conv_out_31 = F.relu(self.nl31(self.conv31(x3)))
        pool_out_31 = self.pool31(conv_out_31)
        conv_out_32 = F.relu(self.nl32(self.conv32(pool_out_31)))
        pool_out_32 = self.pool32(conv_out_32)
        pool_out_32 = pool_out_32.view(-1, 64 * 2 * 2 * 2)
        fc_out_31 = F.relu(self.fc31(pool_out_32))
        dropout_31 = self.dropout31(fc_out_31)
        fc_out_32 = F.relu(self.fc32(dropout_31))

        mid3 = torch.softmax(self.fc33(fc_out_32), dim=1)
        L3 = torch.mean(criterion(mid3, label))
        ################################################################33
        concat1 = torch.cat((fc_out_12, fc_out_22, fc_out_32), dim=1)
        f11 = self.fusion11(concat1)
        f21 = self.fusion21(concat1)
        f31 = self.fusion31(concat1)

        alpha1 = torch.sigmoid(f11)
        alpha2 = torch.sigmoid(f21)
        alpha3 = torch.sigmoid(f31)

        f1 = torch.multiply(alpha1, fc_out_12)
        f2 = torch.multiply(alpha2, fc_out_22)
        f3 = torch.multiply(alpha3, fc_out_32)

        f14 = torch.sum(f1)
        f24 = torch.sum(f2)
        f34 = torch.sum(f3)

        f4 = torch.stack([f14, f24, f34])
        beta = torch.softmax(f4, dim=0)

        f13 = beta[0]
        f23 = beta[1]
        f33 = beta[2]

        ######################### Fusion ################################
        feature_cat = torch.cat((fc_out_12, fc_out_22, fc_out_32), dim=1)
        fc_out_f1 = F.relu(self.f1(feature_cat))
        mid = self.f2(fc_out_f1)
        prediction = torch.softmax(mid, dim=1)

        # return prediction, f13 * L1 + f23 * L2 + f33 * L3
        return prediction

        # # stage 1
        # x1 = F.relu(self.conv11(x1))
        # x1 = self.nl11(x1)
        #
        # x2 = F.relu(self.conv21(x2))
        # x2 = self.nl21(x2)
        #
        # x3 = F.relu(self.conv31(x3))
        # x3 = self.nl31(x3)
        #
        # x1, x2, x3 = self.Co1(x1, x2, x3)
        #
        # x1 = self.pool11(x1)
        # x2 = self.pool21(x2)
        # x3 = self.pool31(x3)
        #
        # # stage 2
        # x1 = F.relu(self.conv12(x1))
        # x1 = self.nl12(x1)
        #
        # x2 = F.relu(self.conv22(x2))
        # x2 = self.nl22(x2)
        #
        # x3 = F.relu(self.conv32(x3))
        # x3 = self.nl32(x3)
        #
        # x1, x2, x3 = self.Co2(x1, x2, x3)
        #
        # x1 = self.pool12(x1)
        # x2 = self.pool22(x2)
        # x3 = self.pool32(x3)
        #
        # # stage 3
        # x1 = F.relu(self.conv13(x1))
        # x1 = self.nl13(x1)
        #
        # x2 = F.relu(self.conv23(x2))
        # x2 = self.nl23(x2)
        #
        # x3 = F.relu(self.conv33(x3))
        # x3 = self.nl33(x3)
        #
        # x1, x2, x3 = self.Co3(x1, x2, x3)
        #
        # x1 = self.pool13(x1)
        # x2 = self.pool23(x2)
        # x3 = self.pool33(x3)
        #
        # # stage 4
        # x1 = x1.view(-1, 64 * 2 * 2 * 2)
        # x1 = F.relu(self.fc11(x1))  # output(1024)
        # x1 = self.dropout11(x1)
        # x1 = F.relu(self.fc12(x1))  # output(32)
        # x1 = self.dropout12(x1)
        #
        # x2 = x2.view(-1, 64 * 2 * 2 * 2)
        # x2 = F.relu(self.fc21(x2))  # output(1024)
        # x2 = self.dropout21(x2)
        # x2 = F.relu(self.fc22(x2))  # output(32)
        # x2 = self.dropout22(x2)
        #
        # x3 = x3.view(-1, 64 * 2 * 2 * 2)
        # x3 = F.relu(self.fc31(x3))  # output(1024)
        # x3 = self.dropout31(x3)
        # x3 = F.relu(self.fc32(x3))  # output(32)
        # x3 = self.dropout32(x3)
        #
        # # stage 5
        # concat1 = torch.cat((x1, x2, x3), dim=1)
        #
        # # ABAF
        # # w_fusion_11 = weight_variable([150, 50])
        # w_fusion_11 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # # b_fusion_11 = bias_variable([50])
        # b_fusion_11 = torch.zeros([50])+0.1
        # w_fusion_21 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # b_fusion_21 = torch.zeros([50])+0.1
        # w_fusion_31 = torch.tensor(truncnorm.rvs(-1,1,size=[150,50]))
        # b_fusion_31 = torch.zeros([50])+0.1
        #
        # w_fusion_12 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_12 = torch.zeros([50]) + 0.1
        # w_fusion_22 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_22 = torch.zeros([50]) + 0.1
        # w_fusion_32 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fusion_32 = torch.zeros([50]) + 0.1
        #
        # f11 = torch.matmul(concat1, w_fusion_11) + b_fusion_11
        # f21 = torch.matmul(concat1, w_fusion_21) + b_fusion_21
        # f31 = torch.matmul(concat1, w_fusion_31) + b_fusion_31
        #
        # alpha1 = torch.sigmoid(f11)
        # alpha2 = torch.sigmoid(f21)
        # alpha3 = torch.sigmoid(f31)
        #
        # f1 = torch.multiply(alpha1, x1)
        # f2 = torch.multiply(alpha2, x2)
        # f3 = torch.multiply(alpha3, x3)
        #
        # concat2 = torch.cat((x1, x2, x3), dim=1)
        # f12 = torch.matmul(concat2, w_fusion_12) + b_fusion_12
        # f22 = torch.matmul(concat2, w_fusion_22) + b_fusion_22
        # f32 = torch.matmul(concat2, w_fusion_32) + b_fusion_32
        #
        # f14 = torch.reduce_sum(torch.sigmoid(f12))
        # f24 = torch.reduce_sum(torch.sigmoid(f22))
        # f34 = torch.reduce_sum(torch.sigmoid(f32))
        #
        # f4 = torch.stack([f14, f24, f34])
        # beta = torch.nn.softmax(f4)
        #
        # f13 = beta[0]
        # f23 = beta[1]
        # f33 = beta[2]
        #
        # ######################################     Fusion      ####################################
        # feature_cat = torch.cat([f1, f2, f3], 1)
        # w_fc_f1 = torch.tensor(truncnorm.rvs(-1, 1, size=[150, 50]))
        # b_fc_f1 = torch.zeros([50]) + 0.1
        # fc_out_f1 = torch.nn.relu(torch.matmul(feature_cat, w_fc_f1) + b_fc_f1)
        #
        # w_fc_f2 = torch.tensor(truncnorm.rvs(-1, 1, size=[50, 2]))
        # b_fc_f2 = torch.zeros([2]) + 0.1
        # mid = torch.matmul(fc_out_f1, w_fc_f2) + b_fc_f2
        # prediction = torch.nn.softmax(mid)
        #
        # x = self.fcfusion(x)  # output(2)
        # return x
