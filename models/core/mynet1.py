import torch.nn as nn
import torch
import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))


class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False) #2048,2048
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)#2048,2048
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False) #2048,512
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False) #512,512
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)#512,512
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)#512,512
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)#512,512
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)#512,512
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))
###????????????????????????????????????????? x_master + x_gdb???????
        #print('x_master:{},x_gpb:{}'.format(x_master.shape,x_gpb.shape))
        out = self.relu(x_master + x_gpb)

        return out

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

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        #fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out




class Mynet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512, 1024]):
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

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        # self.squeeze_excite4 = Squeeze_Excite_Block(filters[3])
        #
        # self.residual_conv4 = ResidualConv(filters[3], filters[4], 2, 1)
        #sencond encoder

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())
        self.output_layer2 = nn.Sequential(nn.Conv2d(2, 1, 1), nn.Sigmoid())


        self.gau_1 = GAU(filters[4], filters[3])
        self.gau_2 = GAU(filters[3], filters[2])
        self.gau_3 = GAU(filters[2], filters[1])
        self.gau_4 = GAU(filters[1], filters[0])

        # self.conv1 = nn.Conv2d(1536,1280,1)
        # self.conv2 = nn.Conv2d(768,640,1)
        # self.conv3 = nn.Conv2d(384,320,1)
        self.conv1 = nn.Conv2d(filters[2]*6, filters[1]*10, 1)
        self.conv2 = nn.Conv2d(filters[1]*6, filters[0]*10, 1)
        self.conv3 = nn.Conv2d(filters[0]*6, 320, 1)




    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # print(x1.shape) ([1, 64, 512, 512])

        x2 = self.squeeze_excite1(x1)
        # print(x2.shape) ([1, 64, 512, 512])
        x2r = self.residual_conv1(x2)
        # print(x2.shape) ([1, 128, 256, 256])

        x3 = self.squeeze_excite2(x2r)
        # print(x3.shape) ([1, 128, 256, 256])
        x3r = self.residual_conv2(x3)
        # print(x3.shape)  ([1, 256, 128, 128])

        x4 = self.squeeze_excite3(x3r)
        # print(x4.shape)  ([1, 256, 128, 128])
        x4r = self.residual_conv3(x4)
        # print(x4.shape) ([1, 512, 64, 64])

        x5 = self.aspp_bridge(x4r)
        # print(x5.shape) ([1, 1024, 64, 64])
        x6 = self.attn1(x3r, x5)
        # print(x6.shape) ([1, 1024, 64, 64])
        x6r = self.upsample1(x6)
        # print(x6.shape) ([1, 1024, 128, 128])
        gau2 = self.gau_2(x4r, x4)  # 1, 256, 128, 128
        merge1 = torch.cat([gau2, x6r, x4], dim=1)  # 1536,128,128
        x6m = self.conv1(merge1)
        # print(x6.shape)
        # x6 = torch.cat([x6, x3r], dim=1)
        # print(x6.shape) ([1, 1280, 128, 128])
        x6ur = self.up_residual_conv1(x6m)
        # print(x6.shape) ([1, 512, 128, 128])

        x7 = self.attn2(x2r, x6ur)
        # print(x7.shape) ([1, 512, 128, 128])
        x7u = self.upsample2(x7)
        # print(x7.shape) ([1, 512, 256, 256])
        gau3 = self.gau_3(x3r, x3)  # 1,128,256
        merge2 = torch.cat([gau3, x7u, x3], dim=1)  # 1,768,256
        x7m = self.conv2(merge2)
        # x7 = torch.cat([x7, x2r], dim=1)
        # print(x7.shape) ([1, 640, 256, 256])
        x7ur = self.up_residual_conv2(x7m)
        # print(x7.shape) ([1, 256, 256, 256])

        x8 = self.attn3(x1, x7ur)
        # print(x8.shape) ([1, 256, 256, 256])
        x8u = self.upsample3(x8)
        # print(x8.shape) ([1, 256, 512, 512])
        gau4 = self.gau_4(x2r, x2)  # 1,64,512
        merge3 = torch.cat([gau4, x8u, x2], dim=1)  # 1,384,512
        x8m = self.conv3(merge3)
        # x8 = torch.cat([x8, x1], dim=1)
        # print(x8.shape) ([1, 320, 512, 512])
        x8ur = self.up_residual_conv3(x8m)
        # print(x8.shape) ([1, 128, 512, 512])

        x9 = self.aspp_out(x8ur)
        # print(x9.shape) ([1, 64, 512, 512])
        out1 = self.output_layer(x9)
        # print(out.shape) ([1, 1, 512, 512])

        input2 = x1 * out1
        # print(input2.shape)
        x2_ = self.squeeze_excite1(input2) #([1, 64, 512, 512])

        x2r_ = self.residual_conv1(x2_)
        # print(x2.shape) ([1, 128, 256, 256])
        x3_ = self.squeeze_excite2(x2r_)
        # print(x3.shape) ([1, 256, 256, 256])
        x3r_ = self.residual_conv2(x3_)
        # print(x3.shape)  ([1, 512, 128, 128])

        x4_ = self.squeeze_excite3(x3r_)  #([1, 256, 128, 128])
        # print(x4.shape)
        x4r_ = self.residual_conv3(x4_)  #([1, 512, 64, 64])
        # print(x4r_.shape)

        x5_ = self.aspp_bridge(x4r_)
        # print(x5.shape) ([1, 1024, 64, 64])
        x6_ = self.attn1(x3r, x5_)
        # print(x6.shape) ([1, 1024, 64, 64])
        x6r_ = self.upsample1(x6_)
        # print(x6.shape) ([1, 1024, 128, 128])
        gau2 = self.gau_2(x4r_, x4_)  # 1, 256, 128, 128
        merge1 = torch.cat([gau2, x6r_, x4_], dim=1)  # 1536,128,128
        x6m_ = self.conv1(merge1)
        # print(x6.shape)
        # x6 = torch.cat([x6, x3r], dim=1)
        # print(x6.shape) ([1, 1280, 128, 128])
        x6ur_ = self.up_residual_conv1(x6m_)
        # print(x6.shape) ([1, 512, 128, 128])

        x7_ = self.attn2(x2r, x6ur_)
        # print(x7.shape) ([1, 512, 128, 128])
        x7u_ = self.upsample2(x7_)
        # print(x7.shape) ([1, 512, 256, 256])
        gau3 = self.gau_3(x3r, x3)  # 1,128,256
        merge2 = torch.cat([gau3, x7u_, x3_], dim=1)  # 1,768,256
        x7m_ = self.conv2(merge2)
        # x7 = torch.cat([x7, x2r], dim=1)
        # print(x7.shape) ([1, 640, 256, 256])
        x7ur_ = self.up_residual_conv2(x7m_)
        # print(x7.shape) ([1, 256, 256, 256])

        x8_ = self.attn3(input2, x7ur_)
        # print(x8.shape) ([1, 256, 256, 256])
        x8u_ = self.upsample3(x8_)
        # print(x8.shape) ([1, 256, 512, 512])
        gau4 = self.gau_4(x2r, x2)  # 1,64,512
        merge3 = torch.cat([gau4, x8u_, x2_], dim=1)  # 1,384,512
        x8m_ = self.conv3(merge3)
        # x8 = torch.cat([x8, x1], dim=1)
        # print(x8.shape) ([1, 320, 512, 512])
        x8ur_ = self.up_residual_conv3(x8m_)
        # print(x8.shape) ([1, 128, 512, 512])

        x9_ = self.aspp_out(x8ur_)
        # print(x9.shape) ([1, 64, 512, 512])
        out2 = self.output_layer(x9_)
        # print(out.shape) ([1, 1, 512, 512])
        out3= torch.cat([out1,out2],dim=1)
        out3 = self.output_layer2(out3)
        # print(out3.shape)


        return out3


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