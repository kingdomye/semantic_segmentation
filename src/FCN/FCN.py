"""
FCN: Fully Convolutional Network
"""

import torch
import torch.nn as nn
from torchvision import models

import os
os.environ["TORCH_HOME"] = "../../checkpoints"


class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()

        # 加载VGG16骨干预训练网络
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        features = list(vgg.features.children())
        self.features_pool3 = nn.Sequential(*features[:17])
        self.features_pool4 = nn.Sequential(*features[17:24])
        self.features_pool5 = nn.Sequential(*features[24:])

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)

        self._initialize_weights()

    def forward(self, x):
        pool3 = self.features_pool3(x)
        pool4 = self.features_pool4(pool3)
        pool5 = self.features_pool5(pool4)

        x = self.fc6(pool5)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        score_fr = self.score_fr(x)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(pool4)

        if upscore2.size() != score_pool4.size():
            upscore2 = torch.nn.functional.interpolate(upscore2, size=score_pool4.shape[2:], mode='bilinear',
                                                       align_corners=False)

        fuse_pool4 = upscore2 + score_pool4
        upscore_pool4 = self.upscore_pool4(fuse_pool4)

        score_pool3 = self.score_pool3(pool3)

        if upscore_pool4.size() != score_pool3.size():
            upscore_pool4 = torch.nn.functional.interpolate(upscore_pool4, size=score_pool3.shape[2:], mode='bilinear',
                                                            align_corners=False)

        fuse_pool3 = upscore_pool4 + score_pool3
        out = self.upscore8(fuse_pool3)

        return out

    def _initialize_weights(self):
        # 仅初始化新增加的层，保护 backbone
        new_layers = [
            self.fc6, self.fc7,
            self.score_fr, self.score_pool4, self.score_pool3,
            self.upscore2, self.upscore8, self.upscore_pool4
        ]
        for layer in new_layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.zero_()


if __name__ == '__main__':
    fcn = FCN8s()
    # 设为评估模式 (这对 Dropout 和 BatchNorm 很重要，虽然 VGG16 只有 Dropout)
    fcn.eval()

    from PIL import Image
    import torchvision.transforms as transforms

    test_input_image_file = '../../outputs/test.png'

    # 1. 强制转换为 RGB (修复报错的核心)
    test_input = Image.open(test_input_image_file).convert('RGB')

    # 2. 预处理
    # VGG16 预训练模型强依赖于标准化的输入 (mean/std)，
    # 如果不加 Normalize，虽然不报错，但输出结果会完全不对。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_input = transform(test_input)

    # 3. 增加 Batch 维度 [C, H, W] -> [1, C, H, W]
    test_input = test_input.unsqueeze(0)

    # 4. 前向传播
    # 使用 no_grad 节省内存
    with torch.no_grad():
        test_output = fcn(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    # 可视化在一张图中
    from matplotlib import pyplot as plt

    # ==========================================
    # 1. 处理输入图 (反归一化)
    # ==========================================
    # 取出 batch 中的第一张图: [1, 3, H, W] -> [3, H, W]
    img_tensor = test_input[0].cpu()

    # 定义归一化参数 (必须和你 transform 中用的一致)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 反归一化: original = normalized * std + mean
    img_vis = img_tensor * std + mean

    # 限制数值在 [0, 1] 之间 (防止 matplotlib 报 clipping 警告)
    img_vis = torch.clamp(img_vis, 0, 1)

    # 转换维度适应 matplotlib: [3, H, W] -> [H, W, 3]
    img_vis = img_vis.permute(1, 2, 0).numpy()

    # ==========================================
    # 2. 处理输出图 (Argmax 获取类别)
    # ==========================================
    # test_output shape: [1, 21, H, W]
    # 在第1个维度(channel)上取最大值的索引 -> 变成 [1, H, W]
    pred_mask = test_output.argmax(dim=1)

    # 降维转 numpy: [1, H, W] -> [H, W]
    pred_mask = pred_mask.squeeze().cpu().numpy()

    # ==========================================
    # 3. 画图 (左右对比)
    # ==========================================
    plt.figure(figsize=(12, 6))

    # 左边画原图
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_vis)
    plt.axis('off')  # 不显示坐标轴

    # 右边画预测结果
    plt.subplot(1, 2, 2)
    plt.title("Prediction Mask")

    # cmap='tab20' 适合显示分类索引 (每个整数一种颜色)
    # vmin=0, vmax=20 确保颜色映射范围固定在 VOC 的 21 个类别内
    plt.imshow(pred_mask, cmap='tab20', vmin=0, vmax=20)
    plt.colorbar(label='Class Index')  # 显示颜色对应的类别ID
    plt.axis('off')

    plt.tight_layout()
    plt.show()
