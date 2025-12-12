"""
FCN: Fully Convolutional Network
"""

import torch
import torch.nn as nn
from torchvision import models
from FCNDataset import VOCSegmentDataset

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
