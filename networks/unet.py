# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.nn import init

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BasicBlock(nn.Module):
    expansion = 1
    '''
    expansion通道扩充比例
    out_channels就是输出的channel
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    expansion = 4

    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # print(f'x4={x4.shape},{x3.shape}')
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output1 = self.out_conv(x)

        # print(f'out:output1={output1.shape},output2={output2.shape}')
        return output1

class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        #Decoder1
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class[0],
                                  kernel_size=3, padding=1)
        self.out_conv_x1 = nn.Conv2d(24, self.n_class[0],
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class[0],
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class[0],
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class[0],
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class[0],
                                      kernel_size=3, padding=1)
        #反卷积，低位特征变成高位特征
        self.trans_conv_layer3 = nn.ConvTranspose2d(self.n_class[0],self.n_class[0],
                                                    kernel_size=8, stride=8)
        self.trans_conv_layer2 = nn.ConvTranspose2d(self.n_class[0], self.n_class[0],
                                                    kernel_size=4, stride=4)
        self.trans_conv_layer1 = nn.ConvTranspose2d(self.n_class[0], self.n_class[0],
                                                    kernel_size=2, stride=2)

        # Decoder2
        self.out_conv_d2 = nn.Conv2d(self.ft_chns[0], self.n_class[1],
                                  kernel_size=3, padding=1)

        self.out_conv_dp4_d2 = nn.Conv2d(self.ft_chns[4], self.n_class[1],
                                      kernel_size=3, padding=1)
        self.out_conv_dp3_d2 = nn.Conv2d(self.ft_chns[3], self.n_class[1],
                                      kernel_size=3, padding=1)
        self.out_conv_dp2_d2 = nn.Conv2d(self.ft_chns[2], self.n_class[1],
                                      kernel_size=3, padding=1)
        self.out_conv_dp1_d2 = nn.Conv2d(self.ft_chns[1], self.n_class[1],
                                      kernel_size=3, padding=1)
        # 反卷积，低位特征变成高位特征
        self.trans_conv_layer3_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
                                                    kernel_size=8, stride=8)
        self.trans_conv_layer2_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
                                                    kernel_size=4, stride=4)
        self.trans_conv_layer1_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
                                                    kernel_size=2, stride=2)
        self.feature_noise = FeatureNoise()

    # def forward(self, feature, shape):
    def forward(self, feature):
        shape=[256,256]
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:   #decoder1
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        # dp3_out_seg = self.trans_conv_layer3(dp3_out_seg)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        if self.training:  # decoder2
            dp3_out_seg_d2 = self.out_conv_dp3_d2(Dropout(x, p=0.5))
        else:
            dp3_out_seg_d2 = self.out_conv_dp3_d2(x)
        # dp3_out_seg_d2 = self.trans_conv_layer3_d2(dp3_out_seg_d2)
        dp3_out_seg_d2 = torch.nn.functional.interpolate(dp3_out_seg_d2, shape)

        x = self.up2(x, x2)
        if self.training: # decoder1
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        # dp2_out_seg = self.trans_conv_layer2(dp2_out_seg)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        if self.training:# decoder2
            dp2_out_seg_d2 = self.out_conv_dp2_d2(FeatureDropout(x))
        else:
            dp2_out_seg_d2 = self.out_conv_dp2_d2(x)
        # dp2_out_seg_d2 = self.trans_conv_layer2_d2(dp2_out_seg_d2)
        dp2_out_seg_d2 = torch.nn.functional.interpolate(dp2_out_seg_d2, shape)

        # dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:# decoder1
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = self.trans_conv_layer1(dp1_out_seg)
        # dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)


        if self.training:# decoder2
            dp1_out_seg_d2 = self.out_conv_dp1_d2(self.feature_noise(x))
        else:
            dp1_out_seg_d2 = self.out_conv_dp1_d2(x)
        dp1_out_seg_d2 = self.trans_conv_layer1_d2(dp1_out_seg_d2)
        # dp1_out_seg_d2 = torch.nn.functional.interpolate(dp1_out_seg_d2, shape)

        x = self.up4(x, x0)

        dp0_out_seg = self.out_conv(x)# decoder1
        dp0_out_seg_d2 = self.out_conv_d2(x)# decoder2


        cat1 = torch.cat([dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg], dim=1)
        cat2 = torch.cat([dp0_out_seg_d2, dp1_out_seg_d2, dp2_out_seg_d2, dp3_out_seg_d2], dim=1)

        return cat1,cat2
        #return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg,dp0_out_seg_d2, dp1_out_seg_d2, dp2_out_seg_d2, dp3_out_seg_d2

class Decoder_Light_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_Light_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.class_conv1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.class_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(128*128*1, 128)

        #Decoder1
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class[0],
                                  kernel_size=3, padding=1)
        self.out_conv_x1 = nn.Conv2d(24, self.n_class[0],
                                  kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class[0],
                                      kernel_size=3, padding=1)
        #反卷积，低位特征变成高位特征
        self.trans_conv_layer3 = nn.ConvTranspose2d(self.n_class[0],self.n_class[0],
                                                    kernel_size=8, stride=8)
        self.trans_conv_layer2 = nn.ConvTranspose2d(self.n_class[0], self.n_class[0],
                                                    kernel_size=4, stride=4)
        self.trans_conv_layer1 = nn.ConvTranspose2d(self.n_class[0], self.n_class[0],
                                                    kernel_size=2, stride=2)

        # Decoder2
        self.out_conv_d2 = nn.Conv2d(self.ft_chns[0], self.n_class[1],
                                  kernel_size=3, padding=1)
        self.out_conv_dp1_d2 = nn.Conv2d(self.ft_chns[1], self.n_class[1],
                                      kernel_size=3, padding=1)
        # # 反卷积，低位特征变成高位特征
        # self.trans_conv_layer3_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
        #                                             kernel_size=8, stride=8)
        # self.trans_conv_layer2_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
        #                                             kernel_size=4, stride=4)
        # self.trans_conv_layer1_d2 = nn.ConvTranspose2d(self.n_class[1], self.n_class[1],
        #                                             kernel_size=2, stride=2)
        self.feature_noise = FeatureNoise()

    # def forward(self, feature, shape):
    def forward(self, feature):
        shape = [256, 256]
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)

        x = self.up2(x, x2)

        x = self.up3(x, x1)
        if self.training:# decoder1
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        # dp1_out_seg = self.trans_conv_layer1(dp1_out_seg)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)


        if self.training:# decoder2
            dp1_out_seg_d2 = self.out_conv_dp1_d2(self.feature_noise(x))
        else:
            dp1_out_seg_d2 = self.out_conv_dp1_d2(x)
        # dp1_out_seg_d2 = self.trans_conv_layer1_d2(dp1_out_seg_d2)
        dp1_out_seg_d2 = torch.nn.functional.interpolate(dp1_out_seg_d2, shape)

        x = self.up4(x, x0)

        dp0_out_seg = self.out_conv(x)# decoder1
        dp0_out_seg_d2 = self.out_conv_d2(x)# decoder2
        # print(x.shape)
        # x = self.class_conv1(x)
        # x = self.class_pool(x)#(20,1,128,128)
        # x = x.view(x.size(0), -1)
        # classification = self.classifier(x)

        cat1 = torch.cat([dp0_out_seg, dp1_out_seg], dim=1)
        cat2 = torch.cat([dp0_out_seg_d2, dp1_out_seg_d2], dim=1)
        # print(f'zhge:',classification.shape)
        # return cat1, cat2, classification
        return cat1, cat2

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.8)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        feature = self.encoder(x)
        d1, d2 = self.decoder(feature)
        return d1,d2

class Light_UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(Light_UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_Light_URPC(params)

    def forward(self, x):
        feature = self.encoder(x)
        d1, d2 = self.decoder(feature)
        return d1, d2