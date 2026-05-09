# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        features = init_features
        # 编码器
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)
        
        # 瓶颈层
        self.bottleneck = self.conv_block(features * 8, features * 16)
        
        # 解码器
        self.dec4 = self.conv_block(features * 16 + features * 8, features * 8)
        self.dec3 = self.conv_block(features * 8 + features * 4, features * 4)
        self.dec2 = self.conv_block(features * 4 + features * 2, features * 2)
        self.dec1 = self.conv_block(features * 2 + features, features)
        
        # 输出层
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #  Padding=1 保证卷积后的特征图尺寸不变
        #  BatchNorm2d:
        #  ReLU 提供非线性，让网络能表示复杂函数

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # 解码路径
        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))
        # interpolate是对图像进行插值运算，模式是双线性插值
        # 输出层
        output = self.final_conv(dec1)
        return output


# 测试模型
if __name__ == '__main__':
    model = UNet()  # 创建模型实例
    # print(model)   
    # 构造一个随机输入张量（假设是1张图像，3通道，512×512）
    input_tensor = torch.randn(1, 3, 512, 512)  # (batch_size=1, channels=3, height=512, width=512)
    output = model(input_tensor)  # 执行前向传播
    print("输出形状:", output.shape)  # 打印输出形状
