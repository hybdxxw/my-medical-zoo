import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from transformers import transformer
import math
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn import functional as F
#ablation for transfomer version


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_MSF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_MSF, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




class Mynet(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(Mynet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.ECAttention = ECAAttention()
        self.transformer1 = transformer.Transformer(num_channels =filters[0],attention_for_seg=True,num_tokens = filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.transformer2 = transformer.Transformer(num_channels=filters[1], attention_for_seg=True, num_tokens=filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.transformer3 = transformer.Transformer(num_channels=filters[2], attention_for_seg=True, num_tokens=filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.squeeze_excite4 = Squeeze_Excite_Block(filters[3])
        self.transformer4 = transformer.Transformer(num_channels=filters[3], attention_for_seg=True, num_tokens=filters[3])
        self.residual_conv4 = ResidualConv(filters[3], filters[4], 2, 1)

        self.MSF1 = RFB_MSF(128,filters[0])
        self.MSF2 = RFB_MSF(256,filters[0])
        self.MSF3 = RFB_MSF(512, filters[0])
        self.aggbridge = aggregation(channel=filters[0])

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # self.ra4_conv5_up = nn.ConvTranspose2d(1, 1, kernel_size=64, stride=32)

        # ---- reverse attention branch 3 ----
        # self.ra4_3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.ra3_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra3_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16)

        # ---- reverse attention branch 2 ----
        # self.ra3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.ra2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra2_conv4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8)
        self._init_weights()






    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)  #([1, 32, 512, 512])
        # print(x1.shape)

        # x2 = self.squeeze_excite1(x1)  #([1, 32, 512, 512])
        x2 = self.ECAttention(x1)
        # print(x2.shape)
        x2 = self.transformer1(x2)
        # print(x2.shape)
        x2r = self.residual_conv1(x2)  #[1, 64, 256, 256])
        # print(x2.shape)

        # x3 = self.squeeze_excite2(x2r) #([1, 64, 256, 256])
        x3 = self.ECAttention(x2r)
        x3 = self.transformer2(x3)
        # print(x3.shape)
        x3r = self.residual_conv2(x3)   #([1, 128, 128, 128])
        # print(x3.shape)

        # x4 = self.squeeze_excite3(x3r)  #([1, 128, 128, 128])
        x4 = self.ECAttention(x3r)
        x4 = self.transformer3(x4)
        # print(x4.shape)
        x4r = self.residual_conv3(x4)  #([1, 256, 64, 64])
        # print(x4r.shape)

        # x5 = self.squeeze_excite4(x4r)  #1,256,64
        x5 = self.ECAttention(x4r)
        x5 = self.transformer4(x5)
        x5r = self.residual_conv4(x5)   #1,512,32
        # print(x5r.shape)
        x3MSF= self.MSF1(x3r) #32,128
        # print(x3MSF.shape)
        x4MSF = self.MSF2(x4r) #32,64
        # print(x4MSF.shape)
        x5MSF = self.MSF3(x5r) #32,32
        # print(x5MSF.shape)
        aggbridge = self.aggbridge(x5MSF,x4MSF,x3MSF)  #1,1,128
        # print(aggbridge.shape)
        # ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(aggbridge, scale_factor=4,mode='bilinear')  # Sup-1 (bs, 1, 128, 128) -> (bs, 1,512, 512)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(aggbridge, scale_factor=0.25, mode='bilinear') # 32,32
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x5r)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=16, mode='bilinear')  # Sup-2 (bs, 1, 32, 32) -> (bs, 1, 512, 512)

        # ---- reverse attention branch_3 ----
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear') # 64
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 256, -1, -1).mul(x4r)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=8, mode='bilinear')
        # lateral_map_3 = self.crop(self.ra3_conv4_up(x), x_size)  # NOTES: Sup-3 (bs, 1, 64, 64) -> (bs, 1, 512, 512)

        # ---- reverse attention branch_2 ----
        # x = self.ra3_2(x)
        # crop_2 = self.crop(x, x2.size())
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')  #128
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 128, -1, -1).mul(x3r)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=4, mode='bilinear')
        # lateral_map_2 = self.crop(self.ra2_conv4_up(x), x_size)  # NOTES: Sup-4 (bs, 1, 128, 128) -> (bs, 1, 512, 512)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





# def test():
#     x=torch.randn((1,3,512,512))
#     model =Mynet(channel=3)
#     predicts =model(x)
#     # assert predicts.shape == x.shape
#     print(x.shape)
#     print(predicts.shape)
#
# if __name__ =="__main__":
#     test()


    #model = LadderNet(inplanes=1, num_classes=2, filters=10).cuda()
    #print(torchsummary.summary(model, input_size=(1, 64, 64),device="cuda"))
    # print(LadderNet(1,2,4,10))
#dummy_input = torch.rand(1, 1, 64, 64).cuda()#????????????1???64*