import torch
import torch.nn as nn
import torch.nn.functional as F

from src.VocDataset import VOCSegmentDataset


class UNet(nn.Module):
    def __init__(self, num_classes=None, in_channels=1):
        super(UNet, self).__init__()

        # 1、Encoder
        # 1*572*572 -> 64*568*568
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64*284*284 -> 128*280*280
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128*140*140 -> 256*136*136
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256*68*68 -> 512*64*64
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512*32*32 -> 1024*32*32
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        # 2、Decoder
        # 1024*28*28 -> 512*56*56
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)

        # 512*52*52 -> 256*104*104
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=1)
        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)

        # 256*100*100 -> 128*200*200
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=1)
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True)

        # 128*196*196 -> 64*388*388
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1)
        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True)

        # 映射到类别个数
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _crop_tensor(tensor, target_tensor):
        tensor_size_h = tensor.size()[2]
        tensor_size_w = tensor.size()[3]

        target_size_h = target_tensor.size()[2]
        target_size_w = target_tensor.size()[3]

        delta_h = (tensor_size_h - target_size_h) // 2
        delta_w = (tensor_size_w - target_size_w) // 2

        tensor = tensor[:, :, delta_h: delta_h + target_size_h, delta_w: delta_w + target_size_w]

        return tensor

    def forward(self, x):
        # 1、Encoder
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        c1 = self.relu1_2(x2)
        down1 = self.max_pool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        c2 = self.relu2_2(x4)
        down2 = self.max_pool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        c3 = self.relu3_2(x6)
        down3 = self.max_pool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        c4 = self.relu4_2(x8)
        down4 = self.max_pool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        c5 = self.relu5_2(x10)

        # UpConv and skip-connection
        # Block 1
        up1 = self.up_conv_1(c5)
        if up1.size() != c4.size():
            up1 = F.interpolate(up1, size=c4.shape[2:], mode='bilinear', align_corners=True)
        up1 = torch.cat([c4, up1], dim=1)
        y1 = self.conv6_1(up1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        # Block 2
        up2 = self.up_conv_2(y2)
        if up2.size() != c3.size():
            up2 = F.interpolate(up2, size=c3.shape[2:], mode='bilinear', align_corners=True)
        up2 = torch.cat([c3, up2], dim=1)
        y3 = self.conv7_1(up2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        # Block 3
        up3 = self.up_conv_3(y4)
        if up3.size() != c2.size():
            up3 = F.interpolate(up3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        up3 = torch.cat([c2, up3], dim=1)
        y5 = self.conv8_1(up3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        # Block 4
        up4 = self.up_conv_4(y6)
        if up4.size() != c1.size():
            up4 = F.interpolate(up4, size=c1.shape[2:], mode='bilinear', align_corners=True)
        up4 = torch.cat([c1, up4], dim=1)
        y7 = self.conv9_1(up4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        out = self.conv10(y8)

        return out


if __name__ == '__main__':
    root = '../../datasets/VOC2012'
    dataset = VOCSegmentDataset(root_dir=root, image_set='train')
    model = UNet(num_classes=21, in_channels=3)

    test_image, test_mask = dataset[0]
    print(test_image.shape)
    print(test_mask.shape)

    output_test = model(test_image.unsqueeze(0))
    print(output_test.shape)

    from matplotlib import pyplot as plt
    pred_mask = torch.argmax(output_test, dim=1).squeeze(0).detach().cpu().numpy()

    # -------------------------------------------------
    # 2. 处理原始图片 (Image)
    # -------------------------------------------------
    # [3, H, W] -> [H, W, 3] 用于 plt 显示
    img_vis = test_image.permute(1, 2, 0).numpy()

    # 如果图片做过归一化(比如减均值除方差)，这里需要简单的反归一化以便显示
    # 这里做一个简单的 min-max 归一化到 0-1 之间，防止显示全黑或全白
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

    # -------------------------------------------------
    # 3. 处理真实标签 (Ground Truth)
    # -------------------------------------------------
    # [1, H, W] -> [H, W]
    gt_mask = test_mask.squeeze(0).numpy()

    # -------------------------------------------------
    # 4. 绘图
    # -------------------------------------------------
    plt.figure(figsize=(15, 5))

    # 图1：原始图片
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img_vis)
    plt.axis('off')

    # 图2：真实标签 (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    # cmap='jet' 或 'tab20' 能更好地区分不同类别
    plt.imshow(gt_mask, cmap='jet', vmin=0, vmax=20)
    plt.axis('off')

    # 图3：模型预测 (Prediction)
    plt.subplot(1, 3, 3)
    plt.title("Model Prediction (Random Weights)")
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
