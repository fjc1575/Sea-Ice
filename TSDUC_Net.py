import torch
from torch import nn



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=2):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 自适应最大池化操作

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CSA(nn.Module):
    def __init__(self, channel):
        super(CSA, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out






class separable_conv_block:
    def __init__(self, ch_in, ch_out):
        super(separable_conv_block, self).__init__()
        # nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1, groups=groups)
        self.depth_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, groups=ch_in),
        self.point_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out,kernel_size=1, stride=1, padding=0, groups=1)


    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=10, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_DCB1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DCB1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=15, dilation=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_DCB2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DCB2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=25, dilation=5, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_DCB3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DCB3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=35, dilation=7, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, padding):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class multi_scaled_dilation_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        super(multi_scaled_dilation_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2 * dilation)),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True, groups= ch_in),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class TSDCU_net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(TSDCU_net,
              self).__init__()

        self.separable_conv = separable_conv_block(img_ch, img_ch)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(img_ch, 16 * 3, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(img_ch, 16 * 3, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(img_ch, 16 * 3, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(img_ch, 16 * 3, kernel_size=11, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(img_ch, 16 * 3, kernel_size=15, dilation=3)

        self.Conv1_1 = conv_block_DCB1(ch_in=16 * 3 * 5, ch_out=8)
        self.Conv1_2 = conv_block_DCB2(ch_in=8, ch_out=8)
        self.Conv1_3 = conv_block_DCB3(ch_in=8, ch_out=8)
        self.Conv2_1 = conv_block_DCB1(ch_in=8, ch_out=16)
        self.Conv2_2 = conv_block_DCB2(ch_in=16, ch_out=16)
        self.Conv2_3 = conv_block_DCB3(ch_in=16, ch_out=16)
        self.Conv3_1 = conv_block_DCB1(ch_in=16, ch_out=32)
        self.Conv3_2 = conv_block_DCB2(ch_in=32, ch_out=32)
        self.Conv3_3 = conv_block_DCB3(ch_in=32, ch_out=32)
        self.Conv4_1 = conv_block_DCB1(ch_in=32, ch_out=64)
        self.Conv4_2 = conv_block_DCB2(ch_in=64, ch_out=64)
        self.Conv4_3 = conv_block_DCB3(ch_in=64, ch_out=64)
        self.Conv5_1 = conv_block_DCB1(ch_in=64, ch_out=128)
        self.Conv5_2 = conv_block_DCB2(ch_in=128, ch_out=128)
        self.Conv5_3 = conv_block_DCB3(ch_in=128, ch_out=128)

        self.o1 = CSA(channel=8)
        self.o2 = CSA(channel=16)
        self.o3 = CSA(channel=32)
        self.o4 = CSA(channel=64)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=False):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        x1_1 = self.Conv1_1(muti_scale_x)
        x1_2 = self.Conv1_2(x1_1)
        x1_3 = self.Conv1_3(x1_2)

        x2 = self.Maxpool(x1_3)

        x2_1 = self.Conv2_1(x2)
        x2_2 = self.Conv2_2(x2_1)
        x2_3 = self.Conv2_3(x2_2)

        x3 = self.Maxpool(x2_3)

        x3_1 = self.Conv3_1(x3)
        x3_2 = self.Conv3_2(x3_1)
        x3_3 = self.Conv3_3(x3_2)

        x4 = self.Maxpool(x3_3)

        x4_1 = self.Conv4_1(x4)
        x4_2 = self.Conv4_2(x4_1)
        x4_3 = self.Conv4_3(x4_2)

        x5 = self.Maxpool(x4_3)

        x5_1 = self.Conv5_1(x5)
        x5_2 = self.Conv5_2(x5_1)
        x5_3 = self.Conv5_3(x5_2)

        o1 = self.o1(x1_3)
        o2 = self.o2(x2_3)
        o3 = self.o3(x3_3)
        o4 = self.o4(x4_3)

        d5 = self.Up5(x5_3)
        d5 = torch.cat((o4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((o3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((o2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((o1, d2), dim=1)
        d2 = self.Up_conv2(d2)


        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)
