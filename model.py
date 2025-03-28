import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):   # Chomp1d 用于裁剪输入的时间维度，去掉多余的填充，以确保时间卷积网络的因果性（即未来的信息不会影响当前的预测）
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size        # chomp_size 是裁剪的大小（通常与卷积的 padding 相等）

    def forward(self, x):       # 定义前向传播逻辑，对输入张量的最后一个维度（即时间维度）进行裁剪
        return x[:, :, :-self.chomp_size].contiguous()  # 从输入张量 x 中取时间维度的前 x.size(2) - chomp_size 部分，去掉多余的填充；.contiguous(): 确保裁剪后的张量在内存中是连续存储的


class TemporalBlock(nn.Module):     # TCN 的基本构造模块
    """
        两个卷积层：负责特征提取
        因果性裁剪（Chomp1d）：保证卷积结果不依赖未来信息
        激活函数和 Dropout：增加非线性和正则化
        残差连接：缓解梯度消失问题，提升训练稳定性
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Args:
            n_inputs:           输入通道数
            n_outputs:          输出通道数
            kernel_size:        卷积核的大小
            stride:             卷积的步幅
            dilation:           扩张率，用于扩展感受野
            padding:            填充大小，确保输出序列长度与输入相同;通常为 (kernel_size - 1) * dilation
            dropout:            Dropout 概率，用于正则化
        """
        super(TemporalBlock, self).__init__()
        # 定义第一个卷积层，并使用 weight_norm 进行权重归一化，帮助加速收敛并稳定训练;及其对应的裁剪、激活和 Dropout 层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 定义第二个卷积层及其对应的裁剪、激活和 Dropout 层，与第一层类似
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将上述所有层（包括卷积、裁剪、激活、Dropout）组合成一个顺序执行的网络
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入通道数和输出通道数不同，使用 1x1 卷积调整输入的通道数
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()   # 用于残差连接后的激活
        self.init_weights()     # 调用权重初始化函数

    def init_weights(self):     # 初始化卷积层的权重，采用均值为 0、标准差为 0.01 的正态分布
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):   # 定义前向传播逻辑，包含主路径（self.net(x)）和残差连接（x 或 self.downsample(x)）的计算，最终通过 ReLU 激活输出
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):       # 堆叠多个 TemporalBlock，通过不同的扩张率捕获长短期依赖
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs:     输入通道数 int
            num_channels:   每层的隐藏通道数 list
            kernel_size:    卷积核尺寸 int
            dropout:        Dropout率 float
        """
        super(TemporalConvNet, self).__init__()
        layers = []     # TCN所有层数
        num_levels = len(num_channels)  # num_channels 的长度，表示网络的层数
        for i in range(num_levels):
            dilation_size = 2 ** i      # 每层的扩张率，按 2 的幂次递增
            in_channels = num_inputs if i == 0 else num_channels[i-1]   # 分别表示每层的输入和输出通道数
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]     # 为每一层创建一个 TemporalBlock，并添加到 layers 列表中

        self.network = nn.Sequential(*layers)   # 将所有 TemporalBlock 组合成一个顺序执行的网络

    def forward(self, x):
        return self.network(x)  # 定义前向传播逻辑，直接将输入通过堆叠的网络


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)      # 时间卷积网络（TCN）的核心部分,通过 TemporalConvNet 实现,由多层一维卷积组成的网络，支持因果卷积和扩张卷积,num_channels 决定了每一层卷积的输出通道数
        self.linear = nn.Linear(num_channels[-1], output_size)      # 添加线性层将 TCN网络 最后一层的输出通道映射到指定的输出目标维度（output_size）

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)  batch_size: 批量大小;seq_len: 时间序列长度;input_size: 每个时间步的输入特征维度
        # TCN expects input shape: (batch_size, num_features, sequence_length)
        # x needs to have dimension (N, C, L) in order to be passed into TCN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)        # 将输入从 (N, L, C) 转换为 (N, C, L) 以适应时间卷积的输入要求；output shape: (N, num_channels[-1], L)；将输出从 (N, C, L) 转回 (N, L, C) 以匹配全连接层的输入；output shape: (N, L, num_channels[-1])
        output = self.linear(output)                                # 全连接层处理每个时间步的特征；output shape: (N, L, output_size)
        return output[:, -1, :]                                     # 仅保留最后一个时间步的输出（适用于序列到值的任务）；output shape: (N, output_size)


if __name__ == "__main__":
    tcn = TCN(4, 2, [150] * 4, 5, 0.25)
    #print(tcn)
    x = torch.randn(1, 200, 4)      # 生成测试数据：1个样本，200个时间步，每个时间步4个特征
    print(x.transpose(1, 2).size())     # transpose用于转置第1和第2维度；输出：torch.Size([1, 4, 200])
    y = tcn(x)
    print(y.size())                 # 输出：torch.Size([1, 2])
    print(y)
