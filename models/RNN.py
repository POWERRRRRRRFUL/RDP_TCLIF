import torch.nn as nn
import torch

class SimpleRNN(nn.Module):
    """用于标准RNN模型的定义
    Args:
        in_dim (int): 输入维度
        hidden_size (int): 隐藏层大小
        num_layers (int): RNN层数
    """
    def __init__(self, in_dim=201, hidden_size=64, num_layers=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        assert x.dim() == 3, "Dimension of x is not correct!"  # x: [batch_size, in_dim, 1]
        x = x.squeeze(-1)  # Reshape to [batch_size, in_dim] for LSTM input
        x, _ = self.rnn(x)
        # print(f"RNN output shape: {x.shape}")  # 打印输出形状以调试
        if x.dim() == 3:
            x = x[:, -1, :]  # 如果输出是三维的，使用最后一个时间步
        elif x.dim() == 2:
            x = x  # 如果已经是二维的，直接使用
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")  # 处理不期望的形状
        x = self.classifier(x)
        return x
