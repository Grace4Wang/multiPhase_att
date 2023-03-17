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


    def forward(self, x1, x2, return_nl_map=False):
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


        
        if return_nl_map:
            return z11, f_div_C11,z21, f_div_C21
        return z11,z21



class CoAtt3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(CoAtt3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


