import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """下采样卷积模块：通过卷积+池化压缩序列长度"""
    def __init__(self, c_in):
        """
        :param c_in: 输入通道数（特征维度）
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,     # 输入通道数
                                  out_channels=c_in,    # 输出通道数（与输入相同）
                                  kernel_size=3,        # 卷积核大小
                                  padding=2,            # 填充数（计算后输出长度：L+2-3+1 = L）
                                  padding_mode='circular')  # 使用Circular Padding保持时序周期性，环形填充（适用于周期性时序数据）
        self.norm = nn.BatchNorm1d(c_in)        # 批归一化（按特征维度归一化）
        self.activation = nn.ELU()              # 激活函数（指数线性单元）
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """输入输出形状：[B, L, C] -> [B, L//2, C]"""
        x = self.downConv(x.permute(0, 2, 1))       # 维度置换 [B,C,L]
        x = self.norm(x)            # 批归一化
        x = self.activation(x)      # 激活函数
        x = self.maxPool(x)         # 最大池化 [B,C,L//2]
        x = x.transpose(1, 2)       # 恢复维度 [B, L//2, C]
        return x


class EncoderLayer(nn.Module):
    """编码器层：多头注意力 + 卷积前馈网络"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        :param attention: 注意力模块（如FullAttention）
        :param d_model:   模型隐藏层维度
        :param d_ff:      前馈网络中间维度（默认4*d_model）
        :param dropout:   Dropout比率
        :param activation: 激活函数类型
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention      # 注意力计算模块
        # 前馈网络：两个1D卷积层实现全连接效果
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)      # 注意力后归一化
        self.norm2 = nn.LayerNorm(d_model)      # 前馈后归一化
        self.dropout = nn.Dropout(dropout)      # 随机失活
        self.activation = F.relu if activation == "relu" else F.gelu    # 激活函数选择（ReLU/GELU）

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,                # 自注意力模式（Q=K=V=x）
            attn_mask=attn_mask,    # 注意力掩码（如因果掩码）
            tau=tau, delta=delta    # 温度参数（某些注意力变体使用），位置相关参数（如Flowformer）
        )
        x = x + self.dropout(new_x)     # 残差连接 + Dropout

        y = x = self.norm1(x)           # 层归一化
        # 前馈网络（维度转换适应1D卷积）
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))   # [B,d_ff,L]
        y = self.dropout(self.conv2(y).transpose(-1, 1))    # [B,L,d_model]

        return self.norm2(x + y), attn  # 残差连接 + 归一化


class Encoder(nn.Module):
    """堆叠编码器层：支持注意力层与卷积层交替"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        :param attn_layers: 注意力层列表（EncoderLayer）
        :param conv_layers:  下采样卷积层列表（ConvLayer）
        :param norm_layer:   最终归一化层
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)       # 注意力层堆叠
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer      # 最终归一化（如LayerNorm）

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """编码过程：序列逐步抽象"""
        # x [B, L, D]
        attns = []      # 收集各层注意力矩阵（可视化用）
        # 若有卷积下采样层（如Informer的蒸馏结构）
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None       # 仅第一层使用delta（位置相关参数）
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)      # 注意力计算
                x = conv_layer(x)       # 下采样（序列长度减半）
                attns.append(attn)      # 记录注意力
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)  # 处理最后一层（无对应卷积层）
            attns.append(attn)
        else:       # 无卷积层的纯注意力堆叠（如标准Transformer）
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:       # 最终归一化
            x = self.norm(x)

        return x, attns     # 返回编码结果和注意力矩阵列表


class DecoderLayer(nn.Module):
    """解码器层：自注意力 + 交叉注意力 + 前馈网络"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        :param self_attention: 自注意力模块（掩码）
        :param cross_attention: 交叉注意力模块（无掩码）
        :param d_model: 隐藏层维度
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention        # 自注意力（解码器自回归）
        self.cross_attention = cross_attention      # 交叉注意力（连接编码输出）
        # 前馈网络（与编码器结构相同）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)      # 自注意力后
        self.norm2 = nn.LayerNorm(d_model)      # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)      # 前馈网络后
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """处理流程：自回归解码 -> 编码器信息融合 -> 前馈"""
        # 自注意力（带掩码防止信息泄漏）
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])       # 取输出部分
        x = self.norm1(x)
        # 交叉注意力（查询来自解码器，键值来自编码器）
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        y = x = self.norm2(x)
        # 前馈网络
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """解码器堆叠：多解码层 + 输出投影"""
    def __init__(self, layers, norm_layer=None, projection=None):
        """
        :param layers: 多个DecoderLayer组成的列表
        :param norm_layer: 最终归一化层
        :param projection: 输出投影层（如Linear）
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection        # 输出维度映射（如vocab_size）

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """解码流程：逐层处理 -> 归一化 -> 投影"""
        for layer in self.layers:       # 遍历所有解码层
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:       # 最终归一化
            x = self.norm(x)

        if self.projection is not None:     # 输出投影（如生成词概率）
            x = self.projection(x)
        return x        # [B, L, output_dim]
