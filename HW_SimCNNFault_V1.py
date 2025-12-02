import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ConvAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_pool = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)
        y = self.conv1(avg_pool)
        y = F.relu(y)
        y = self.conv2(y)
        scale = self.sigmoid(y)
        return x * scale + x


class HW_SimCNNFault_V1(nn.Module):
    """
    带多尺度 3×3/5×5 分支 + 重参数化的 1D-CNN 故障诊断网络，
    不再使用 3000×256 这种大 FC。

    输入:
        - (B, 3000)
        - 或 (B, 1, 3000)
    """

    def __init__(self, input_dim: int = 3000, num_classes: int = 13, in_channels: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 重参数化开关
        self.deploy = False
        self.fused_conv: Optional[nn.Conv1d] = None

        # 1. Stem：下采样 + 提升通道
        # 3000 -> 1500 左右
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        # 2. 多尺度多分支（训练形态：多分支）
        # 训练时：conv3 + conv5 + 各自 BN，再相加
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False)
        self.bn5 = nn.BatchNorm1d(32)

        # 3. 后续卷积块，进一步降采样 + 提升通道
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 4. 通道注意力
        self.attention = ConvAttention(64)

        # 5. 全局池化 + 小 MLP 输出头
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # 支持 (B, L) 或 (B, C, L)
        if x.dim() == 2:
            # (B, L) -> (B, 1, L)
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != self.in_channels:
            # 若给成 (B, L, C)，则转成 (B, C, L)
            x = x.permute(0, 2, 1)

        z = self.stem(x)  # (B, 16, L1)

        # 多分支 / 部署形态
        if self.deploy and self.fused_conv is not None:
            # 推理形态：单一 5×5 卷积（conv3+conv5+BN 已融合）
            z = F.relu(self.fused_conv(z))
        else:
            # 训练形态：3×3 分支 + 5×5 分支
            z = self.bn3(self.conv3(z)) + self.bn5(self.conv5(z))
            z = F.relu(z)  # (B, 32, L1)

        # 后续卷积块
        z = self.block2(z)   # (B, 64, L2)
        z = self.block3(z)   # (B, 64, L3)

        # 注意力模块
        z = self.attention(z)  # (B, 64, L3)

        # 全局平均池化 + 小 MLP 头
        z = self.global_pool(z).squeeze(-1)  # (B, 64)
        logits = self.fc_out(z)              # (B, num_classes)

        return logits

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
        """Fuse Conv1d + BatchNorm1d into an equivalent Conv1d kernel/bias."""
        w = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(var + eps)

        w_fused = w * (gamma / std).reshape(-1, 1, 1)
        if conv.bias is not None:
            b_fused = beta + (conv.bias - mean) * gamma / std
        else:
            b_fused = beta - mean * gamma / std
        return w_fused, b_fused

    def switch_to_deploy(self):
        """
        将 3x3+BN 和 5x5+BN 融合为单一 5x5 卷积，供推理加速/简化 C++ 实现。
        训练/加载权重后调用一次，forward 即走 fused 分支。
        """
        if self.deploy:
            return

        k3, b3 = self._fuse_conv_bn(self.conv3, self.bn3)
        k5, b5 = self._fuse_conv_bn(self.conv5, self.bn5)
        pad = (5 - 3) // 2  # 将 3x3 填充到 5x5 以便求和
        k3_padded = F.pad(k3, [pad, pad])

        fused_kernel = k5 + k3_padded
        fused_bias = b5 + b3

        self.fused_conv = nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=True)
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        self.deploy = True
        # 删除冗余分支，节省显存/防止误用
        del self.conv3, self.bn3, self.conv5, self.bn5


__all__ = ["HW_SimCNNFault_V1"]
