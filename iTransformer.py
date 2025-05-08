import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        """
        初始化模型
        :param configs: 配置参数对象，需包含以下属性：
            - seq_len: 输入序列长度（历史时间步数）
            - pred_len: 预测序列长度（未来时间步数）
            - d_model: 嵌入维度（通常256-512）
            - n_heads: 多头注意力头数
            - e_layers: 编码器层数
            - dropout: Dropout概率
            - use_norm: 是否启用归一化
            - output_attention: 是否输出注意力权重
            - embed: 嵌入类型（如'timeF'等）
            - freq: 时间特征频率（如'h'小时，'t'分钟）
        """
        super(Model, self).__init__()
        # 基础参数
        self.seq_len = configs.seq_len      # 输入序列长度（L）
        self.pred_len = configs.pred_len    # 预测序列长度（S）
        self.output_attention = configs.output_attention    # 是否输出注意力矩阵
        self.use_norm = configs.use_norm    # 是否使用序列归一化
        # Embedding
        # 倒置嵌入层（核心组件）
        # 输入形状：[Batch, L, N] -> [Batch, N, E]
        # L:序列长度，N:变量数，E:嵌入维度
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len,    # 输入维度（此处为序列长度，因倒置嵌入）
                                                    configs.d_model,    # 嵌入维度
                                                    configs.embed,      # 嵌入类型（如固定位置编码）
                                                    configs.freq,       # 时间特征频率
                                                    configs.dropout)    # Dropout概率
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        # 编码器架构（堆叠多个EncoderLayer）
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 注意力层（使用FullAttention）
                    AttentionLayer(
                        FullAttention(False,                               # 是否使用因果掩码（此处关闭）
                                      configs.factor,                               # ProbAttention采样因子
                                      attention_dropout=configs.dropout,            # 注意力矩阵Dropout
                                      output_attention=configs.output_attention     # 是否输出注意力
                        ),
                        configs.d_model,        # 输入维度
                        configs.n_heads         # 注意力头数
                    ),
                    configs.d_model,            # 输入/输出维度
                    configs.d_ff,               # 前馈网络隐层维度（默认d_model*4）
                    dropout=configs.dropout,    # Dropout概率
                    activation=configs.activation       # 激活函数（如'relu'）
                ) for l in range(configs.e_layers)      # 堆叠e_layers层
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)      # 层归一化
        )
        # 投影层：将编码后的特征映射到预测长度
        # 输入：[Batch, N, E] -> 输出：[Batch, N, S]
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        前向预测过程
        :param x_enc: 输入序列 [Batch, L, N]（L=seq_len, N=变量数）
        :param x_mark_enc: 时间特征 [Batch, L, D_time]（如月、日、时等）
        :return dec_out: 预测结果 [Batch, S, N]
        """
        # 归一化（来自Non-stationary Transformer）
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # 计算均值和标准差（沿时间维度L）
            means = x_enc.mean(1, keepdim=True).detach()        # [B, 1, N]
            x_enc = x_enc - means       # 去均值化
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)    # [B, 1, N]
            x_enc /= stdev              # 标准化

        _, _, N = x_enc.shape   # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # 倒置嵌入：将变量维度N映射为嵌入维度E
        # 输入：[B, L, N] -> [B, N, E]（传统Transformer为[B, L, E]）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        # 编码器处理：通过多层注意力+前馈网络
        # 输出保持 [B, N, E]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        # 投影到预测长度：将嵌入维度E转换为pred_len S
        # [B, N, E] -> [B, N, S] -> [B, S, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]    # filter the covariates

        # 反归一化（若启用）
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))       # 恢复方差
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))       # 恢复均值

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播入口
        :param x_enc: 历史序列 [B, L, N]
        :param x_mark_enc: 历史时间特征 [B, L, D_time]
        :param x_dec: 解码器输入（占位符，本模型未使用）
        :param x_mark_dec: 解码器时间特征（占位符）
        :return: 预测结果 [B, S, N] 或 (预测结果, 注意力矩阵)
        """
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 根据配置决定是否返回注意力矩阵
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns        # 取最后pred_len个时间步
        else:
            return dec_out[:, -self.pred_len:, :]               # 输出形状[B, L, D]

