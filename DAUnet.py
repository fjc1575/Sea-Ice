import torch
from torch import nn
from torch.nn import Conv2d, Softmax
from torch.nn import functional as F


class Zeropadding(nn.Module):
    def __init__(self):
        super(Zeropadding, self).__init__()
        self.zero = nn.ZeroPad2d(padding=(3, 3, 3, 3))

    def forward(self, input):
        return self.zero(input)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 实现上采样， 指定输出为输入的多少倍数
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),  # padding ：填充，1，就是在上下左右四个方向补一圈0。
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)


class Residual_1(nn.Module):
    '''
    定义普通残差模块
    resnet34为普通残差块，resnet50为瓶颈结构
    '''
    def __init__(self, inchannel, outchannel, stride=1, padding=1, shortcut=None):
        super(Residual_1, self).__init__()
        #resblock的首层，首层如果跨维度，卷积stride=2，shortcut需要1*1卷积扩维
        if inchannel != outchannel:
            stride= 2
            shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )

        # 定义残差块的左部分
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, 3, 1, padding, bias=False),
            nn.BatchNorm2d(outchannel),\
        )

        #定义右部分
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return F.relu(out)




class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DAUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DAUnet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size ：表示做最大池化的窗口大小;stride ：步长
        self.zero = Zeropadding()
        self.conv1 = Conv(in_ch, 64, 7, 2, 0)

        self.R1_1 = Residual_1(64, 64)
        self.R1_2 = Residual_1(64, 64)
        self.R1_3 = Residual_1(64, 64)

        self.R2_1 = Residual_1(64, 128)
        self.R2_2 = Residual_1(128, 128)
        self.R2_3 = Residual_1(128, 128)
        self.R2_4 = Residual_1(128, 128)

        self.R3_1 = Residual_1(128, 256)
        self.R3_2 = Residual_1(256, 256)
        self.R3_3 = Residual_1(256, 256)
        self.R3_4 = Residual_1(256, 256)
        self.R3_5 = Residual_1(256, 256)
        self.R3_6 = Residual_1(256, 256)

        self.R3_6_c = Conv(256, 512, 1, 1, 0)

        self.R4_1 = Residual_1(512, 512)
        self.R4_2 = Residual_1(512, 512)
        self.R4_3 = Residual_1(512, 512)

        self.PAM = PAM_Module(512)
        self.convp = Conv(512, 512, 3, 1, 1)
        self.CAM = CAM_Module(512)
        self.convc = Conv(512, 512, 3, 1, 1)

        self.convcp = Conv(512, 512, 3, 1, 1)

        self.up4 = up_conv(512, 512)
        self.conv4_1 = Conv(640, 256, 3, 1, 1)
        self.conv4_2 = Conv(256, 256, 3, 1, 1)

        self.up3 = up_conv(256, 256)
        self.conv3_1 = Conv(320, 128, 3, 1, 1)
        self.conv3_2 = Conv(128, 128, 3, 1, 1)

        self.up2 = up_conv(128, 128)
        self.conv2_1 = Conv(192, 64, 3, 1, 1)
        self.conv2_2 = Conv(64, 64, 3, 1, 1)

        self.up1 = up_conv(64, 64)
        self.conv1_1 = Conv(64, 32, 3, 1, 1)
        self.conv1_2 = Conv(32, 32, 3, 1, 1)
        self.conv1_3 = Conv(32, 16, 3, 1, 1)
        self.conv1_4 = Conv(16, 16, 3, 1, 1)

        self.conv1_5 = nn.Conv2d(16, out_ch, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.zero(x)
        x1 = self.conv1(x)
        p1 = self.Maxpool(x1)

        R1_1 = self.R1_1(p1)
        R1_2 = self.R1_2(R1_1)
        R1_3 = self.R1_3(R1_2)

        R2_1 = self.R2_1(R1_3)
        R2_2 = self.R2_2(R2_1)
        R2_3 = self.R2_3(R2_2)
        R2_4 = self.R2_4(R2_3)

        R3_1 = self.R3_1(R2_4)
        R3_2 = self.R3_2(R3_1)
        R3_3 = self.R3_3(R3_2)
        R3_4 = self.R3_4(R3_3)
        R3_5 = self.R3_5(R3_4)
        R3_6 = self.R3_6(R3_5)

        R3_6_c = self.R3_6_c(R3_6)
        R4_1 = self.R4_1(R3_6_c)
        R4_2 = self.R4_2(R4_1)
        R4_3 = self.R4_3(R4_2)

        PAM = self.PAM(R4_3)
        PAM = self.convp(PAM)
        CAM = self.CAM(R4_3)
        CAM = self.convc(CAM)
        CP = self.convcp(PAM+CAM)
        CP_UP = self.up4(CP)
        up4_c = torch.cat([R2_4, CP_UP], dim=1)
        up4_c_1 = self.conv4_1(up4_c)
        up4_c_2 = self.conv4_2(up4_c_1)

        up3 = self.up3(up4_c_2)
        up3_c = torch.cat((R1_3, up3), dim=1)
        up3_1 = self.conv3_1(up3_c)
        up3_2 = self.conv3_2(up3_1)

        up2 = self.up2(up3_2)
        up2_c = torch.cat((x1, up2), dim=1)
        up2_1 = self.conv2_1(up2_c)
        up2_2 = self.conv2_2(up2_1)

        up1 = self.up1(up2_2)
        up1_1 = self.conv1_1(up1)
        up1_2 = self.conv1_2(up1_1)
        up1_3 = self.conv1_3(up1_2)
        up1_4 = self.conv1_4(up1_3)
        up1_5 = self.conv1_5(up1_4)

        out = self.sigmoid(up1_5)
        return out





