import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    """用于标准CNN模型的定义
    Args:
        in_dim (int): 输入维度
    """
    def __init__(self, in_dim=201):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),  # 输入通道设为1
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * in_dim, 128),  # 使用 in_dim 作为输入尺寸
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        assert x.dim() == 3, "Dimension of x is not correct!"  # x: [batch_size, in_dim, 1]
        x = x.permute(0, 2, 1)  # 调整维度为 [batch_size, 1, in_dim] 以适应 Conv1d
        output = self.features(x)
        return output
