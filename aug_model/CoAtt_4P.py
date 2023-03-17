import torch
from torch import nn

'''
参考文章”Attention guided discriminative feature learning and adaptive fusion for grading hepatocellular 
carcinoma with Contrast-enhanced MR“ 做的CoAtt模块
代码参考：https://github.com/Lsx0802/discriminative
原文DOI:10.1016/j.compmedimag.2022.102050
'''


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None,sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        ################################ modal 1 #############################
        conv_nd1 = nn.Conv3d
        max_pool_layer1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn1 = nn.BatchNorm3d

        if bn_layer:
            self.W11 = nn.Sequential(
                conv_nd1(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn1(self.in_channels)
            )
            nn.init.constant_(self.W11[1].weight, 0)
            nn.init.constant_(self.W11[1].bias, 0)
        else:
            self.W11 = conv_nd1(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W11.weight, 0)
            nn.init.constant_(self.W11.bias, 0)
            
        if bn_layer:
            self.W12 = nn.Sequential(
                conv_nd1(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn1(self.in_channels)
            )
            nn.init.constant_(self.W12[1].weight, 0)
            nn.init.constant_(self.W12[1].bias, 0)
        else:
            self.W12 = conv_nd1(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W12.weight, 0)
            nn.init.constant_(self.W12.bias, 0)

        self.g1 = conv_nd1(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta1 = conv_nd1(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi1 = conv_nd1(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g1 = nn.Sequential(self.g1, max_pool_layer1)
            self.phi1 = nn.Sequential(self.phi1, max_pool_layer1)
            
            
        ######################################## modal 2 ##########################
        
        conv_nd2 = nn.Conv3d
        max_pool_layer2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn2 = nn.BatchNorm3d


        if bn_layer:
            self.W21 = nn.Sequential(
                conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn2(self.in_channels)
            )
            nn.init.constant_(self.W21[1].weight, 0)
            nn.init.constant_(self.W21[1].bias, 0)
        else:
            self.W21 = conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W21.weight, 0)
            nn.init.constant_(self.W21.bias, 0)
        
        if bn_layer:
            self.W22 = nn.Sequential(
                conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn2(self.in_channels)
            )
            nn.init.constant_(self.W22[1].weight, 0)
            nn.init.constant_(self.W22[1].bias, 0)
        else:
            self.W22 = conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W22.weight, 0)
            nn.init.constant_(self.W22.bias, 0)

        self.g2 = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.theta2 = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi2 = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g2 = nn.Sequential(self.g2, max_pool_layer2)
            self.phi2 = nn.Sequential(self.phi2, max_pool_layer2)

        ######################################## modal 3 ##########################

        conv_nd3 = nn.Conv3d
        max_pool_layer3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn3 = nn.BatchNorm3d

        if bn_layer:
            self.W31 = nn.Sequential(
                conv_nd3(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn3(self.in_channels)
            )
            nn.init.constant_(self.W31[1].weight, 0)
            nn.init.constant_(self.W31[1].bias, 0)
        else:
            self.W31 = conv_nd3(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W31.weight, 0)
            nn.init.constant_(self.W31.bias, 0)
            
        if bn_layer:
            self.W32 = nn.Sequential(
                conv_nd3(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn3(self.in_channels)
            )
            nn.init.constant_(self.W32[1].weight, 0)
            nn.init.constant_(self.W32[1].bias, 0)
        else:
            self.W32 = conv_nd3(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W32.weight, 0)
            nn.init.constant_(self.W32.bias, 0)

        self.g3 = conv_nd3(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta3 = conv_nd3(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi3 = conv_nd3(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g3 = nn.Sequential(self.g3, max_pool_layer3)
            self.phi3 = nn.Sequential(self.phi3, max_pool_layer3)

        ######################################## modal 4 ##########################

        conv_nd4 = nn.Conv3d
        max_pool_layer4 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn4 = nn.BatchNorm3d

        if bn_layer:
            self.W41 = nn.Sequential(
                conv_nd4(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn4(self.in_channels)
            )
            nn.init.constant_(self.W41[1].weight, 0)
            nn.init.constant_(self.W41[1].bias, 0)
        else:
            self.W41 = conv_nd4(in_channels=self.inter_channels, out_channels=self.in_channels,
                                kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W41.weight, 0)
            nn.init.constant_(self.W41.bias, 0)

        if bn_layer:
            self.W42 = nn.Sequential(
                conv_nd4(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn4(self.in_channels)
            )
            nn.init.constant_(self.W42[1].weight, 0)
            nn.init.constant_(self.W42[1].bias, 0)
        else:
            self.W42 = conv_nd2(in_channels=self.inter_channels, out_channels=self.in_channels,
                                kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W42.weight, 0)
            nn.init.constant_(self.W42.bias, 0)

        self.g4 = conv_nd4(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta4 = conv_nd4(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi4 = conv_nd4(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g4 = nn.Sequential(self.g4, max_pool_layer4)
            self.phi4 = nn.Sequential(self.phi4, max_pool_layer4)

    def forward(self, x1, x2, x3, x4, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        ######################################## modal 1 ##########################
        batch_size1 = x1.size(0)

        g_x1 = self.g1(x1).view(batch_size1, self.inter_channels, -1)
        g_x1= g_x1.permute(0, 2, 1)

        theta_x1 = self.theta1(x1).view(batch_size1, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi1(x1).view(batch_size1, self.inter_channels, -1)

        ######################################## modal 2 ##########################
        batch_size2 = x2.size(0)

        g_x2 = self.g2(x2).view(batch_size2, self.inter_channels, -1)
        g_x2= g_x2.permute(0, 2, 1)

        theta_x2 = self.theta2(x2).view(batch_size2, self.inter_channels, -1)
        theta_x2 = theta_x2.permute(0, 2, 1)
        phi_x2 = self.phi2(x2).view(batch_size2, self.inter_channels, -1)
        
        ######################################## modal 3 ##########################
        batch_size3 = x3.size(0)

        g_x3 = self.g3(x3).view(batch_size3, self.inter_channels, -1)
        g_x3= g_x3.permute(0, 2, 1)

        theta_x3 = self.theta3(x3).view(batch_size3, self.inter_channels, -1)
        theta_x3 = theta_x3.permute(0, 2, 1)
        phi_x3 = self.phi3(x3).view(batch_size3, self.inter_channels, -1)

        ######################################## modal 4 ##########################
        batch_size4 = x4.size(0)

        g_x4 = self.g4(x4).view(batch_size4, self.inter_channels, -1)
        g_x4= g_x4.permute(0, 2, 1)

        theta_x4 = self.theta4(x4).view(batch_size4, self.inter_channels, -1)
        theta_x4 = theta_x4.permute(0, 2, 1)
        phi_x4 = self.phi4(x4).view(batch_size4, self.inter_channels, -1)

        ######################################## modal 1 为核心的 CoAtt ##########################
        # modal 12
        f11 = torch.matmul(theta_x1, phi_x2)
        N11= f11.size(-1)
        f_div_C11 = f11 / N11

        y11 = torch.matmul(f_div_C11, g_x2)
        y11 = y11.permute(0, 2, 1).contiguous()
        y11 = y11.view(batch_size1, self.inter_channels, *x1.size()[2:])
        W_y11 = self.W11(y11)
        z11 = W_y11 + x1

        # modal 13
        f12 = torch.matmul(theta_x1, phi_x3)
        N12= f12.size(-1)
        f_div_C12 = f12 / N12

        y12 = torch.matmul(f_div_C12, g_x3)
        y12 = y12.permute(0, 2, 1).contiguous()
        y12 = y12.view(batch_size1, self.inter_channels, *x1.size()[2:])
        W_y12 = self.W12(y12)
        z12 = W_y12 + z11


        ######################################## modal 2 为核心的 CoAtt ##########################
        # modal 21
        f21 = torch.matmul(theta_x2, phi_x1)
        N21= f21.size(-1)
        f_div_C21 = f21 / N21

        y21 = torch.matmul(f_div_C21, g_x1)
        y21 = y21.permute(0, 2, 1).contiguous()
        y21 = y21.view(batch_size2, self.inter_channels, *x2.size()[2:])
        W_y21 = self.W21(y21)
        z21 = W_y21 + x2

        # modal 23
        f22 = torch.matmul(theta_x2, phi_x3)
        N22= f22.size(-1)
        f_div_C22 = f22 / N22

        y22 = torch.matmul(f_div_C22, g_x3)
        y22 = y22.permute(0, 2, 1).contiguous()
        y22 = y22.view(batch_size2, self.inter_channels, *x2.size()[2:])
        W_y22 = self.W22(y22)
        z22 = W_y22 + z21

        ######################################## modal 3 为核心的 CoAtt ##########################
        # modal 31
        f31 = torch.matmul(theta_x3, phi_x1)
        N31 = f31.size(-1)
        f_div_C31 = f31 / N31

        y31 = torch.matmul(f_div_C31, g_x1)
        y31 = y31.permute(0, 2, 1).contiguous()
        y31 = y31.view(batch_size3, self.inter_channels, *x3.size()[2:])
        W_y31 = self.W31(y31)
        z31 = W_y31 + x3

        # modal 32
        f32 = torch.matmul(theta_x3, phi_x4)
        N32 = f32.size(-1)
        f_div_C32 = f32 / N32

        y32 = torch.matmul(f_div_C32, g_x4)
        y32 = y32.permute(0, 2, 1).contiguous()
        y32 = y32.view(batch_size3, self.inter_channels, *x3.size()[2:])
        W_y32 = self.W32(y32)
        z32 = W_y32 + z31
        
        ######################################## modal 4 为核心的 CoAtt ##########################
        # modal 41
        f41 = torch.matmul(theta_x4, phi_x1)
        N41 = f41.size(-1)
        f_div_C41 = f41 / N41

        y41 = torch.matmul(f_div_C41, g_x1)
        y41 = y41.permute(0, 2, 1).contiguous()
        y41 = y41.view(batch_size4, self.inter_channels, *x4.size()[2:])
        W_y41 = self.W41(y41)
        z41 = W_y41 + x4

        # modal 42
        # 这一行不确定对不对，是phi_x2还是phi_x3
        f42 = torch.matmul(theta_x4, phi_x3)
        N42 = f42.size(-1)
        f_div_C42 = f42 / N42

        y42 = torch.matmul(f_div_C42, g_x3)
        y42 = y42.permute(0, 2, 1).contiguous()
        y42 = y42.view(batch_size4, self.inter_channels, *x4.size()[2:])
        W_y42 = self.W42(y42)
        z42 = W_y42 + z41
        
        if return_nl_map:
            return z12, f_div_C11,f_div_C12,z22, f_div_C21,f_div_C22,z32, f_div_C31,f_div_C32, z42, f_div_C41, f_div_C42
        return z12,z22,z32,z42



class CoAtt3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(CoAtt3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


