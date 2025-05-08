import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """经典Transformer位置编码（sin/cos函数形式）"""
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: 嵌入维度（需与模型主维度一致）
        :param max_len: 预计算的最大序列长度（缓存位置编码）
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False     # 固定编码，不参与梯度更新

        # 位置向量：[0,1,...,max_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)        # [max_len,1]
        # 频率项计算：exp(-2i/d_model * ln(10000))
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()     # [d_model//2]

        # 填充奇偶位置：sin(position * div_term) 和 cos(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)        # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)        # 奇数位置

        # 扩展为[1, max_len, d_model]并注册为缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """返回截断到输入序列长度的位置编码
        :param x: 输入张量 [B, L, ...]
        :return: [1, L, d_model]
        """
        return self.pe[:, :x.size(1)]       # 自动广播到batch维度


class TokenEmbedding(nn.Module):
    """时间序列值嵌入（使用1D卷积）"""
    def __init__(self, c_in, d_model):
        """
        :param c_in:  输入通道数（变量数）
        :param d_model: 输出嵌入维度
        """
        super(TokenEmbedding, self).__init__()
        # 根据PyTorch版本选择padding参数（v1.5+支持'circular'模式的padding计算）
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 1D卷积层配置（kernel=3，circular padding）
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)     # 环形填充，适用于周期性序列
        # 参数初始化：He初始化配合LeakyReLU
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')     # 保持前向传播方差，与后续可能的激活函数匹配

    def forward(self, x):
        """输入形状转换：[B, L, C] -> [B, C, L] -> [B, d_model, L] -> [B, L, d_model]"""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """固定式嵌入（不可训练的位置编码）"""
    def __init__(self, c_in, d_model):
        """
        :param c_in: 词汇表大小（时间特征类别数）
        :param d_model: 嵌入维度
        """
        super(FixedEmbedding, self).__init__()

        # 初始化权重矩阵 [c_in, d_model]
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # 位置向量（离散位置索引）
        position = torch.arange(0, c_in).float().unsqueeze(1)
        # 与PositionalEmbedding相同的频率计算
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 填充奇偶位置
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 注册为Embedding层参数
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """返回detach后的嵌入结果，确保不参与梯度计算"""
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """多粒度时间特征嵌入（小时/日/月等）"""
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        :param d_model: 嵌入维度
        :param embed_type: 嵌入类型（'fixed'使用预定义编码，否则可训练）
        :param freq: 时间序列频率（'h'小时，'t'分钟等）
        """
        super(TemporalEmbedding, self).__init__()
        # 各时间特征的类别数定义
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # 选择嵌入层类型
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 按需构建各粒度嵌入层
        if freq == 't':     # 分钟级数据需要分钟嵌入
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """输入x: [B, L, 5]（假设时间特征包含月、日、周、时、分）"""
        x = x.long()        # 确保索引为整数
        # 各时间特征嵌入（维度需匹配才能相加）
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x      # 特征融合


class TimeFeatureEmbedding(nn.Module):
    """连续时间特征嵌入（线性变换）"""
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # 频率到特征维度的映射
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]      # 获取输入特征维度
        self.embed = nn.Linear(d_inp, d_model, bias=False)      # 线性投影

    def forward(self, x):
        return self.embed(x)        # [B, L, d_inp] -> [B, L, d_model]


class DataEmbedding(nn.Module):
    """标准数据嵌入（值+位置+时间特征）"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        :param c_in: 输入变量数
        :param d_model: 嵌入维度
        :param embed_type: 时间嵌入类型（'fixed'或'timeF'）
        :param freq: 时间特征粒度
        """
        super(DataEmbedding, self).__init__()
        # 值嵌入（卷积层）
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 时间特征嵌入（离散或连续）
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # 正则化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        :param x: 值序列 [B, L, C]
        :param x_mark: 时间特征 [B, L, D_time]
        """
        if x_mark is None:      # 无时间特征模式
            x = self.value_embedding(x) + self.position_embedding(x)
        else:                   # 三部分嵌入相加
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)  # 输出正则化


class DataEmbedding_inverted(nn.Module):
    """倒置嵌入（iTransformer专用，将变量维度作为嵌入维度）"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        :param c_in: 输入变量数（处理后成为序列长度维度）
        :param d_model: 嵌入维度
        """
        super(DataEmbedding_inverted, self).__init__()
        # 核心操作：线性变换 [C_in -> d_model]
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """维度倒置处理
        :param x: 输入 [B, L, C] -> 转置为 [B, C, L]
        :param x_mark: 时间特征 [B, L, D] -> 转置为 [B, D, L]
        """
        x = x.permute(0, 2, 1)      # [B, C, L]
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)     # [B, C, d_model]
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            # 拼接变量特征和时间特征（扩展变量维度）
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))        # [B, C+D, L]
        # x: [Batch Variate d_model]
        return self.dropout(x)      # [B, C, d_model]

