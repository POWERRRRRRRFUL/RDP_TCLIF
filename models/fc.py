import torch.nn as nn
import torch
from spiking_neuron import base, neuron
from surrogate import Triangle as SG


class RecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, element_wise_function=lambda x, y: x + y, step_mode='s', hid_dim=64):
        super().__init__()
        self.hid_weight = nn.Linear(hid_dim, hid_dim)
        # nn.init.orthogonal_(self.hid_weight.weight)
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory('y', None)

    def forward(self, x: torch.Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.hid_weight(self.y), x))
        return self.y

    def extra_repr(self) -> str:
        return f'element-wise function={self.element_wise_function}, step_mode={self.step_mode}'


class ff(nn.Module):
    def __init__(self, in_dim=201, spiking_neuron=None):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 64),
                   spiking_neuron()]
        layers += [nn.Linear(64, 256),
                   spiking_neuron()]
        layers += [nn.Linear(256, 256),
                   spiking_neuron()]
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)
        self.in_dim = in_dim

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 201, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx+self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, x.size(1)-self.in_dim:x.size(1), :].reshape(-1, self.in_dim)
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)

class fb(nn.Module):
    def __init__(self, in_dim=201, spiking_neuron=None):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, 64),
                   RecurrentContainer(spiking_neuron(v_threshold=1.0), hid_dim=64)]
        layers += [nn.Linear(64, 256),
                   RecurrentContainer(spiking_neuron(v_threshold=1.0), hid_dim=256)]
        layers += [nn.Linear(256, 256),
                   spiking_neuron(v_threshold=1.0)]
        layers += [nn.Linear(256, 10)]
        self.features = nn.Sequential(*layers)
        self.in_dim = in_dim

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # x: [bs, 201, 1]
        output_current = []
        for time in range(x.size(1)):  # T loop
            start_idx = time
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx+self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, x.size(1)-self.in_dim:x.size(1), :].reshape(-1, self.in_dim)
            output_current.append(self.features(x_t))
        res = torch.stack(output_current, 0)
        return res.sum(0)
class SpikingResBlock(nn.Module):
    """一个用于SNN的基本残差块
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        spiking_neuron (nn.Module): 尖峰神经元模型
        downsample (nn.Module, optional): 用于下采样的层，如果输入和输出通道不一致需要使用。默认为None。
    """
    def __init__(self, in_channels, out_channels, spiking_neuron, downsample=None):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)  # 第一层线性层
        self.sn1 = spiking_neuron()  # 第一层尖峰神经元
        self.conv2 = nn.Linear(out_channels, out_channels)  # 第二层线性层
        self.sn2 = spiking_neuron()  # 第二层尖峰神经元
        self.downsample = downsample  # 用于维度匹配的下采样层

    def forward(self, x):
        identity = x  # 保存输入以进行残差连接
        out = self.conv1(x)  # 通过第一层线性层
        out = self.sn1(out)  # 通过第一层尖峰神经元
        out = self.conv2(out)  # 通过第二层线性层
        out = self.sn2(out)  # 通过第二层尖峰神经元
        if self.downsample:
            identity = self.downsample(x)  # 如果需要下采样，调整输入的维度
        out += identity  # 残差连接
        return out


class ResNet(nn.Module):
    """用于尖峰神经网络的ResNet模型，包含时间注意力机制"""

    def __init__(self, in_dim=201, spiking_neuron=None, num_heads=8):
        super().__init__()
        self.in_dim = in_dim  # 输入维度
        self.num_heads = num_heads  # 注意力机制的头数

        # 初始线性层和尖峰神经元
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 64),
            spiking_neuron()
        )

        # 两个残差块
        self.res_block1 = SpikingResBlock(64, 64, spiking_neuron)
        self.res_block2 = SpikingResBlock(64, 128, spiking_neuron, downsample=nn.Linear(64, 128))

        # 添加自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=self.num_heads)

        # 输出分类层
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # 确保输入维度正确 [bs, 201, 1]
        output_current = []  # 用于存储每个时间步的输出

        for time in range(x.size(1)):  # 按时间步循环
            start_idx = time
            # 根据时间步裁剪输入
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx + self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, x.size(1) - self.in_dim:x.size(1), :].reshape(-1, self.in_dim)

            # 初始层
            x_t = self.layer1(x_t)
            x_t = self.res_block1(x_t)  # 第一个残差块
            x_t = self.res_block2(x_t)  # 第二个残差块

            # 使用自注意力机制
            x_t = x_t.unsqueeze(0)  # 由于MultiheadAttention要求输入有batch_first维度
            x_t_attended, _ = self.attention(x_t, x_t, x_t)  # query, key, value 都是x_t
            x_t_attended = x_t_attended.squeeze(0)  # 去掉多余的维度

            output_current.append(self.fc(x_t_attended))  # 输出分类层

        res = torch.stack(output_current, 0)  # 堆叠时间步的输出
        return res.sum(0)  # 返回时间步输出的和


class AlexNet(nn.Module):
    """用于尖峰神经网络的AlexNet模型
    Args:
        in_dim (int): 输入维度
        spiking_neuron (nn.Module): 尖峰神经元模型
    """
    def __init__(self, in_dim=201, spiking_neuron=None):
        super().__init__()
        # 特征提取部分，由多个线性层和尖峰神经元组成
        self.features = nn.Sequential(
            nn.Linear(in_dim, 64),
            spiking_neuron(),
            nn.Linear(64, 128),
            spiking_neuron(),
            nn.Linear(128, 256),
            spiking_neuron(),
        )
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            spiking_neuron(),
            nn.Linear(256, 10)
        )
        self.in_dim = in_dim

    def forward(self, x):
        assert x.dim() == 3, "dimension of x is not correct!"  # 确保输入维度正确 [bs, 201, 1]
        output_current = []  # 用于存储每个时间步的输出
        for time in range(x.size(1)):  # 按时间步循环
            start_idx = time
            # 根据时间步裁剪输入
            if start_idx < (x.size(1) - self.in_dim):
                x_t = x[:, start_idx:start_idx + self.in_dim, :].reshape(-1, self.in_dim)
            else:
                x_t = x[:, x.size(1) - self.in_dim:x.size(1), :].reshape(-1, self.in_dim)
            x_t = self.features(x_t)  # 特征提取部分
            output_current.append(self.classifier(x_t))  # 分类部分
        res = torch.stack(output_current, 0)  # 堆叠时间步的输出
        return res.sum(0)  # 返回时间步输出的和



