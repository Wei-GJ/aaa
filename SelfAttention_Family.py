import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange        # 用于张量维度操作


# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    """流式注意力机制，通过核函数重加权实现高效计算"""
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)        # 注意力矩阵Dropout

    def kernel_method(self, x):
        """核函数：将输入映射到非线性空间（此处使用Sigmoid）"""
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 调整维度顺序 [B, L, H, E] -> [B, H, L, E]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        # 应用核函数变换
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        # 计算行、列归一化因子（防止除零错误）
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        # 重加权过程：通过两次矩阵乘法优化注意力分布
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))        # 列方向Softmax
        # competition and allocation
        # 竞争与分配机制
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))       # 行方向Sigmoid激活
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis # 列方向Softmax
        # multiply
        # 矩阵乘法计算最终输出
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None      # 返回注意力输出和空矩阵（兼容接口）


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    """内存优化的分块注意力计算，适用于长序列"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale                              # 缩放因子
        self.mask_flag = mask_flag                      # 是否启用掩码
        self.output_attention = output_attention        # 是否输出注意力矩阵
        self.dropout = nn.Dropout(attention_dropout)    # Dropout层

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32     # 分块大小（根据GPU内存调整）
        NEG_INF = -1e10     # -infinity # 负无穷（用于掩码）
        EPSILON = 1e-10     # 防除零小量
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        # 初始化输出和中间变量
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]                # 归一化因子
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF       # 最大值缓存

        # 转移数据到GPU（假设已启用CUDA）
        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        # 分块策略
        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        # 将Q/K/V分割为块
        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)      # 查询块数
        Tc = len(K_BLOCKS)      # 键值块数

        # 分割输出和中间变量
        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        # 分块计算注意力
        for j in range(Tc):     # 遍历键值块
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):     # 遍历查询块
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                # 缩放点积计算
                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale
                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                # 掩码处理
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')     # 调整维度用于广播
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)
                # 稳定化的Softmax计算
                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)
                # 增量更新输出
                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new
        # 合并所有块的结果
        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 调整维度后调用分块计算
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None       # 恢复原始维度


class FullAttention(nn.Module):
    """标准缩放点积注意力机制"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale                              # 缩放因子
        self.mask_flag = mask_flag                      # 是否使用因果掩码
        self.output_attention = output_attention        # 是否输出注意力矩阵
        self.dropout = nn.Dropout(attention_dropout)    # Dropout层

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape      # Batch, 序列长, 头数, 特征维度
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)      # 自动确定缩放因子
        # 计算注意力分数 [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # 应用因果掩码（解码器自注意力）
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        # Softmax归一化
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 值加权求和
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    """概率稀疏注意力，通过采样减少计算量"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor            # 采样因子（控制稀疏度）
        self.scale = scale
        self.mask_flag = mask_flag      # 是否使用掩码
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """采样计算Top-K查询"""
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 随机采样键向量
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))      # 随机采样位置
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        # 计算查询重要性分数
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)      # 稀疏度测量
        M_top = M.topk(n_top, sorted=False)[1]          # 选择Top-K查询索引

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """注意力机制适配层，将输入投影到多头空间"""
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)         # 每个头的键维度
        d_values = d_values or (d_model // n_heads)     # 每个头的值维度

        self.inner_attention = attention                # 具体注意力实现（如FullAttention）
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)        # 输出投影
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # 投影到多头空间 [B, L, H, dh]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # 执行注意力计算
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        # 合并多头输出 [B, L, H*dh] -> [B, L, D]
        out = out.view(B, L, -1)

        return self.out_projection(out), attn       # 输出投影到模型维度


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

