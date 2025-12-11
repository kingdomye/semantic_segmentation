import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request


# ==========================================
# 1. 准备图片
# ==========================================
def load_image(img_path=None):
    """
    加载图片，如果路径为空则下载一张示例图片
    """
    if img_path is None:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        img_path = "dog.jpg"
        try:
            urllib.request.urlretrieve(url, img_path)
        except:
            print("无法下载图片，请确保网络通畅或手动指定本地图片路径。")
            return None

    img = Image.open(img_path).convert('RGB')
    return img


# ==========================================
# 2. 定义卷积处理函数
# ==========================================
def apply_convolution(img_tensor, dilation_rate):
    """
    定义并应用一个卷积层
    """
    # 定义卷积层
    # in_channels=3 (RGB), out_channels=1 (输出单通道灰度特征图以便观察)
    # kernel_size=3
    # padding设置为 dilation_rate，这样可以保持输出图片的尺寸与输入一致 (Same Padding)
    conv = nn.Conv2d(in_channels=3,
                     out_channels=1,
                     kernel_size=3,
                     dilation=dilation_rate,
                     padding=dilation_rate,
                     bias=False)

    # -------------------------------------------------
    # 关键步骤：手动设置卷积核权重 (Laplacian 边缘检测算子)
    # 这样我们能看清膨胀卷积具体的物理意义，而不是看随机噪声
    # -------------------------------------------------
    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=torch.float32)
    normal_kernel = torch.tensor(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]], dtype=torch.float32)

    # 将 3x3 的核扩展为 [out_c, in_c, k, k] -> [1, 3, 3, 3]
    # 对RGB三个通道都应用同样的核，并求和
    kernel = normal_kernel.expand(1, 3, 3, 3)

    # 赋值给卷积层
    conv.weight.data = kernel

    # 前向传播
    with torch.no_grad():
        output = conv(img_tensor)

    return output


# ==========================================
# 3. 主程序
# ==========================================

# A. 加载并预处理图片
img = load_image()  # 如果你有本地图片，填入路径，如: load_image('my_photo.jpg')

if img:
    # 预处理：Resize -> ToTensor (归一化到0-1, 维度变为 C,H,W)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度: [1, 3, 512, 512]

    # B. 执行卷积
    # 1. 普通卷积 (Dilation = 1)
    out_normal = apply_convolution(img_tensor, dilation_rate=1)

    # 2. 膨胀卷积 (Dilation = 4)
    # 膨胀率为4意味着卷积核的像素点之间隔了3个空洞
    out_dilated = apply_convolution(img_tensor, dilation_rate=4)


    # C. 可视化处理
    def tensor_to_img(tensor):
        # 去掉 batch 维度 [1, 1, H, W] -> [H, W]
        img_np = tensor.squeeze().numpy()
        # 取绝对值（边缘检测会有负值），并归一化以便显示
        img_np = np.abs(img_np)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        return img_np


    plt.figure(figsize=(15, 5))

    # 显示原图
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    # 显示普通卷积
    plt.subplot(1, 3, 2)
    plt.title("Standard Conv (dilation=1)\nFine Edges")
    plt.imshow(tensor_to_img(out_normal), cmap='gray')
    plt.axis('off')

    # 显示膨胀卷积
    plt.subplot(1, 3, 3)
    plt.title("Dilated Conv (dilation=4)\nLarger Receptive Field")
    plt.imshow(tensor_to_img(out_dilated), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
