import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1*dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_out = self.conv_s1(x)
        x_out = self.bn_s1(x_out)
        x_out = self.relu_s1(x_out)

        return x_out


def _upsample(x, target):

    return F.upsample(x, size=target.shape[2:], mode='bilinear')


class RSU7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU7, self).__init__()

        self.re_bn_conv_in = REBNCONV(in_channels=in_channels, out_channels=out_channels, dirate=1)

        self.re_bn_conv1 = REBNCONV(in_channels=out_channels, out_channels=mid_channels, dirate=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv2 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv3 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv4 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv5 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv6 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv7 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=2)

        self.re_bn_conv6d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv5d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv4d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv3d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv1d = REBNCONV(in_channels=mid_channels*2, out_channels=out_channels, dirate=1)

    def forward(self, x):
        x_in = self.re_bn_conv_in(x)

        x1 = self.re_bn_conv1(x_in)
        x1 = self.max_pool1(x1)

        x2 = self.re_bn_conv2(x1)
        x2 = self.max_pool2(x2)

        x3 = self.re_bn_conv3(x2)
        x3 = self.max_pool3(x3)

        x4 = self.re_bn_conv4(x3)
        x4 = self.max_pool4(x4)

        x5 = self.re_bn_conv5(x4)
        x5 = self.max_pool5(x5)

        x6 = self.re_bn_conv6(x5)

        x7 = self.re_bn_conv7(x6)

        x6d = self.re_bn_conv6d(torch.cat((x7, x6), dim=1))
        x6d_up = _upsample(x6d, x5)

        x5d = self.re_bn_conv5d(torch.cat((x6d_up, x5), dim=1))
        x5d_up = _upsample(x5d, x4)

        x4d = self.re_bn_conv4d(torch.cat((x5d_up, x4), dim=1))
        x4d_up = _upsample(x4d, x3)

        x3d = self.re_bn_conv3d(torch.cat((x4d_up, x3), dim=1))
        x3d_up = _upsample(x3d, x2)

        x2d = self.re_bn_conv2d(torch.cat((x3d_up, x2), dim=1))
        x2d_up = _upsample(x2d, x1)

        x1d = self.re_bn_conv1d(torch.cat((x2d_up, x1), dim=1))

        out = x1d + x_in

        return out


class RSU6(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU6, self).__init__()

        self.re_bn_conv_in = REBNCONV(in_channels=in_channels, out_channels=out_channels, dirate=1)

        self.re_bn_conv1 = REBNCONV(in_channels=out_channels, out_channels=mid_channels, dirate=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv2 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv3 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv4 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv5 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv6 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=2)

        self.re_bn_conv5d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv4d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv3d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv1d = REBNCONV(in_channels=mid_channels*2, out_channels=out_channels, dirate=1)

    def forward(self, x):
        x_in = self.re_bn_conv_in(x)

        x1 = self.re_bn_conv1(x_in)
        x1 = self.max_pool1(x1)

        x2 = self.re_bn_conv2(x1)
        x2 = self.max_pool2(x2)

        x3 = self.re_bn_conv3(x2)
        x3 = self.max_pool3(x3)

        x4 = self.re_bn_conv4(x3)
        x4 = self.max_pool4(x4)

        x5 = self.re_bn_conv5(x4)

        x6 = self.re_bn_conv6(x5)

        x5d = self.re_bn_conv5d(torch.cat((x6, x5), dim=1))
        x5d_up = _upsample(x5d, x4)

        x4d = self.re_bn_conv4d(torch.cat((x5d_up, x4), dim=1))
        x4d_up = _upsample(x4d, x3)

        x3d = self.re_bn_conv3d(torch.cat((x4d_up, x3), dim=1))
        x3d_up = _upsample(x3d, x2)

        x2d = self.re_bn_conv2d(torch.cat((x3d_up, x2), dim=1))
        x2d_up = _upsample(x2d, x1)

        x1d = self.re_bn_conv1d(torch.cat((x2d_up, x1), dim=1))

        out = x1d + x_in

        return out


class RSU5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU5, self).__init__()

        self.re_bn_conv_in = REBNCONV(in_channels=in_channels, out_channels=out_channels, dirate=1)

        self.re_bn_conv1 = REBNCONV(in_channels=out_channels, out_channels=mid_channels, dirate=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv2 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv3 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv4 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv5 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv4d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv3d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv1d = REBNCONV(in_channels=mid_channels*2, out_channels=out_channels, dirate=1)

    def forward(self, x):
        x_in = self.re_bn_conv_in(x)

        x1 = self.re_bn_conv1(x_in)
        x1 = self.max_pool1(x1)

        x2 = self.re_bn_conv2(x1)
        x2 = self.max_pool2(x2)

        x3 = self.re_bn_conv3(x2)
        x3 = self.max_pool3(x3)

        x4 = self.re_bn_conv4(x3)
        x4 = self.max_pool4(x4)

        x5 = self.re_bn_conv5(x4)

        x4d = self.re_bn_conv4d(torch.cat((x5, x4), dim=1))
        x4d_up = _upsample(x4d, x3)

        x3d = self.re_bn_conv3d(torch.cat((x4d_up, x3), dim=1))
        x3d_up = _upsample(x3d, x2)

        x2d = self.re_bn_conv2d(torch.cat((x3d_up, x2), dim=1))
        x2d_up = _upsample(x2d, x1)

        x1d = self.re_bn_conv1d(torch.cat((x2d_up, x1), dim=1))

        out = x1d + x_in

        return out


class RSU4(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU4, self).__init__()

        self.re_bn_conv_in = REBNCONV(in_channels=in_channels, out_channels=out_channels, dirate=1)

        self.re_bn_conv1 = REBNCONV(in_channels=out_channels, out_channels=mid_channels, dirate=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv2 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv3 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.re_bn_conv4 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv3d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv1d = REBNCONV(in_channels=mid_channels*2, out_channels=out_channels, dirate=1)

    def forward(self, x):
        x_in = self.re_bn_conv_in(x)

        x1 = self.re_bn_conv1(x_in)
        x1 = self.max_pool1(x1)

        x2 = self.re_bn_conv2(x1)
        x2 = self.max_pool2(x2)

        x3 = self.re_bn_conv3(x2)
        x3 = self.max_pool3(x3)

        x4 = self.re_bn_conv4(x3)

        x3d = self.re_bn_conv3d(torch.cat((x4, x3), dim=1))
        x3d_up = _upsample(x3d, x1)

        x2d = self.re_bn_conv2d(torch.cat((x3d_up, x2), dim=1))
        x2d_up = _upsample(x2d, x_in)

        x1d = self.re_bn_conv1d(torch.cat((x2d_up, x1), dim=1))

        out = x1d + x_in

        return out


class RSU4F(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(RSU4F, self).__init__()

        self.re_bn_conv_in = REBNCONV(in_channels=in_channels, out_channels=out_channels, dirate=1)

        self.re_bn_conv1 = REBNCONV(in_channels=out_channels, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)
        self.re_bn_conv3 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv4 = REBNCONV(in_channels=mid_channels, out_channels=mid_channels, dirate=1)

        self.re_bn_conv3d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv2d = REBNCONV(in_channels=mid_channels*2, out_channels=mid_channels, dirate=1)
        self.re_bn_conv1d = REBNCONV(in_channels=mid_channels*2, out_channels=out_channels, dirate=1)

    def forward(self, x):
        x_in = self.re_bn_conv_in(x)
        x1 = self.re_bn_conv1(x_in)
        x2 = self.re_bn_conv2(x1)
        x3 = self.re_bn_conv3(x2)
        x4 = self.re_bn_conv4(x3)

        x3d = self.re_bn_conv3d(torch.cat((x4, x3), dim=1))
        x2d = self.re_bn_conv2d(torch.cat((x3d, x2), dim=1))
        x1d = self.re_bn_conv1d(torch.cat((x2d, x1), dim=1))

        out = x1d + x_in

        return out


class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(U2Net, self).__init__()

        # ENCODER
        self.stage1 = RSU7(in_channels=in_channels, mid_channels=32, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(in_channels=64, mid_channels=32, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(in_channels=128, mid_channels=64, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(in_channels=256, mid_channels=128, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(in_channels=512, mid_channels=256, out_channels=512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(in_channels=512, mid_channels=256, out_channels=512)

        # DECODER
        self.stage5d = RSU4F(in_channels=1024, mid_channels=256, out_channels=512)
        self.stage4d = RSU4(in_channels=1024, mid_channels=128, out_channels=256)
        self.stage3d = RSU5(in_channels=512, mid_channels=64, out_channels=128)
        self.stage2d = RSU6(in_channels=256, mid_channels=32, out_channels=64)
        self.stage1d = RSU7(in_channels=128, mid_channels=16, out_channels=32)

        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(in_channels=6*in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x1 = self.pool1(x1)

        x2 = self.stage2(x1)
        x2 = self.pool2(x2)

        x3 = self.stage3(x2)
        x3 = self.pool3(x3)

        x4 = self.stage4(x3)
        x4 = self.pool4(x4)

        x5 = self.stage5(x4)
        x5 = self.pool5(x5)

        x6 = self.stage6(x5)
        x6_up = _upsample(x6, x5)

        x5d = self.stage5d(torch.cat((x6_up, x5), dim=1))
        x5d_up = _upsample(x5d, x4)

        x4d = self.stage4d(torch.cat((x5d_up, x4), dim=1))
        x4d_up = _upsample(x4d, x3)

        x3d = self.stage3d(torch.cat((x4d_up, x3), dim=1))
        x3d_up = _upsample(x3d, x2)

        x2d = self.stage2d(torch.cat((x3d_up, x2), dim=1))
        x2d_up = _upsample(x2d, x1)

        x1d = self.stage1d(torch.cat((x2d_up, x1), dim=1))

        side1 = self.side1(x1d)

        side2 = self.side2(x2d)
        side2 = _upsample(side2, side1)

        side3 = self.side3(x3d)
        side3 = _upsample(side3, side2)

        side4 = self.side4(x4d)
        side4 = _upsample(side4, side3)

        side5 = self.side5(x5d)
        side5 = _upsample(side5, side4)

        side6 = self.side6(x6_up)
        side6 = _upsample(side6, side5)

        side0 = self.out_conv(torch.cat((side1, side2, side3, side4, side5, side6), dim=1))

        return F.sigmoid(side0), F.sigmoid(side1), F.sigmoid(side2), F.sigmoid(side3), F.sigmoid(side4), F.sigmoid(side5), F.sigmoid(side6)
