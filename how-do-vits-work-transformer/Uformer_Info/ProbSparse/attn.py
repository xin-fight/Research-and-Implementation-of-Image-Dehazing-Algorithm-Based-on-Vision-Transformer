import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import numpy as np

from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # B, L_q, H, E
        _, S, _, D = values.shape  # B, L_v/k, H, E
        scale = self.scale or 1. / sqrt(E)

        # blhe->bhle   bshe->bhes  相乘： bhle * bges -> bhls
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #
        #     # masked_fill_：用value填充tensor中与mask中值为1位置相对应的元素
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # bhls    bshd->bhsd    相乘：bhls * bhsd -> bhld -> blhd
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention)

    Appendix E Reproducibility:
        1. ProbSparse Self-Attention首先对 Key 进行采样，得到U_part个K_sample，对每个 query 关于K_sample求M值
        2. 找到 M值最大的 u个 query ，对这Top-u个 query 关于所有Key求score值
        3. 其中 Q^ 是Top-u的 query 组成的矩阵，对于没有被选中的那些 query 的score值取 mean(V)
        * _get_initial_context()函数中，先将S所有行都置成 mean(V)
        * _update_context()函数将那些Top-u中的行的scoreh值更新为 attention值
    """

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        :param mask_flag: Encoder False/ Decoder True
        :param factor: Probsparse attn factor (defaults to 5)
        :param scale: 对Q^ * K^T 的放缩，默认None，表示使用1. / sqrt(D)
        :param attention_dropout: The probability of dropout (defaults to 0.05)
        :param output_attention: Whether to output attention in encoder(defaults to False)
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)  # 定义softmax

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        (queries, keys, sample_k=U_part, n_top=u)
        :param Q: Q
        :param K: K
        :param sample_k: U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        :param n_top:         u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        :return:  M_top找出前top_u的query对应的索引           [B, H, top_u]
                  Q_K: use the reduced Q to calculate Q_K   [B, H, M_top, L_K]
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        """calculate the sampled Q_K"""
        # unsqueeze(-3)-> [B, H, 1, L_K, E]
        # .expand：扩大维度 [B, H, 1, L_K, E] -> [B, H, L_Q, L_K, E]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # torch.randint: 数值随机在0-L_K之间     [L_Q, sample_k]
        # 对每个query而言，从K中选择sample_k个key
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q

        """对K_expand的L_Q, L_K 维度进行索引， 结果为对每个 query 选出了 K_sample"""
        # torch.arange.unsqueeze -> [L_Q, 1] 值为: 0-L_Q
        # index_sample：[L_Q, sample_k]
        # [B, H, L_Q, sample_k, E]
        """
            整数数组索引，参考：https://www.runoob.com/numpy/numpy-advanced-indexing.html
            ([[0],        ([[1, 2],                 维度广播    ([[0，0],                 ([[1, 2],
              [1]]) L_Q*1   [2, 1]]) L_Q*sample_k   ------->     [1，1]]) L_Q*sample_k     [2, 1]]) L_Q*sample_k
                   索引为:  [0,1]  [1,2] 
                           [0,2]  [1,1]
        """
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        """Q_K_sample 对应Appendix E Reproducibility流程图中的 S^ = Q*K_sample^T"""
        # Q.unsqueeze：[B, H, L_Q, 1, E]
        # K_sample.transpose -> [B, H, L_Q, E, sample_k]
        # matmul -> [B, H, L_Q, 1, sample_k] -> Q_K_sample：[B, H, L_Q, sample_k]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        """find the Top_k query with sparisty measurement"""
        # Q_K_sample: [B, H, L_Q, sample_k]
        # max(-1)[0] 找最后一个维度最大的值  ([0]表示值，[1]表示该维度上索引)   -> [B, H, L_Q]
        # Q_K_sample.sum(-1)   -> [B, H, L_Q]
        # 公式4的计算过程   M: [B, H, L_Q]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        """找出前top_u的索引"""
        # 默认在最后一个维度找最大的几个值  ([0]表示值，[1]表示该维度上索引)
        # [B, H, L_Q] -> [B, H, top_u]
        M_top = M.topk(n_top, sorted=False)[1]

        """find reduced Q and use the reduced Q to calculate Q_K"""
        # torch.arange(B) -> [B, 1, 1]  0-B-1
        # torch.arange(H) -> [1, H, 1]  0-H-1
        # 与上面的整数数组索引相同: Q的第一维是从0-B-1，第二维是0-H-1, 第三维是M_top（表示取M_top中索引位置对应值）, 最后一维取全部
        # Q_reduce：[B, H, L_Q, E] -> [B, H, M_top, E]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        """
            Q = [[[ 0,  1,  2],  1*4*3维
                  [ 3,  4,  5],
                  [ 6,  7,  8],
                  [ 9, 10, 11]]]
            
            M_top = torch.tensor([[[  0,  1,  2],[  2,  1,  0],[  2,  2,  2],[  1,  1,  1]]])  1*4*3维
            Q[torch.arange(1)[:, None, None],  torch.arange(4)[None, :, None],  M_top]
            第一维取0，第二维取0-4，然而：第三维M_top给出了1*4*3维的数，每个数对应原矩阵的第三维，因此范围都是[0, 2]
                                        举例来说M_top中 M_top[0, 3, 1] == 1, 表示取 Q[0, 3, 1] == 10
        """

        # Q_reduce: [B, H, M_top, E]
        # K.transpose: [B, H, E, L_K]
        # Q_K: [B, H, M_top, L_K]
        """
            注意：是Q_reduce与全部的K做运算
        """
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        _get_initial_context()函数中，先将S所有行都置成 mean(V)
        :param V:   [B, H, L_V, D]
        :param L_Q: Q
        :return: [B, H, L_Q, D]
        """
        B, H, L_V, D = V.shape

        # Encoder False/ Decoder True
        if not self.mask_flag:  # Encoder
            # V_sum = V.sum(dim=-2)

            # V_sum: [B, H, L_V, D] -> [B, H, D]
            V_sum = V.mean(dim=-2)

            # unsqueeze -> [B, H, 1, D]
            # contex: [B, H, L_Q, D]  将均值复制L_Q次
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, relative_position_bias, SW_mask, attn_mask):
        """
         _update_context()函数将那些Top-u中的行的scoreh值更新为 attention值
         事实上，输入都为x时，L_Q == L_K == L_V = N
        :param context_in: 已将所有行都置成 mean(V) [B, H, L_Q, D]
        :param V: Value  [B, H, L_V, D]
        :param scores: Q^ * K^T / sqrt(D)    [B, H, M_top, L_K]
        :param index: 找出前top_u的query对应的索引   [B, H, top_u]
        :param L_Q:
        :param relative_position_bias: [nH, N=Mh*Mw, N=Mh*Mw]
        :param SW_mask: [nW, Mh*Mw, Mh*Mw]
        :param attn_mask:
        :return: [B, H, L_Q, D]  [B, H, L_V, L_V] 值为 1/L_V   数据类型转换
        """
        B, H, L_V, D = V.shape

        # [B, H, M_top, L_K==N]  只有部分q与全部K的值
        attn = torch.softmax(scores, dim=-1)  # .Softmax(dim=-1)(scores)

        """### 让relative_position_bias与全部的 Q*K 相加 而不是仅仅和 部分q组成的 Q^*K 相加 """
        # .size(-1): 取最后一维的维度
        # 原代码：attn: [B=batch_size*num_windows, num_heads, N=Mh*Mw, N=Mh*Mw]
        # attn: [B, H, M_top, L_K == N] -> N
        # relative_position_bias: [nH, Mh*Mw, Mh*Mw] -> Mh*Mw
        """
            但是，在LeWinTransformerBlock类中有：  (对比SwinTransformerBlock中是pad feature maps to multiples of window size)
            if min(self.input_resolution) <= self.win_size:
                self.shift_size = 0
                self.win_size = min(self.input_resolution)
            attn的Mh,Mw是由输入x的第二个维度N所给  
        """
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)  # 在最后一个维度重复

        ##########################################################
        '''
            按照公式中的q @ k.transpose加入相对位置偏执
            代码中会直接得到attention的结果context_in, 而attn是Q*K只有部分的结果; 所以改进：只对部分结果attn加上 **对应的** Bias
            由上：
                原：attn:[batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
                原：relative_position_bias:             [nH, Mh*Mw, Mh*Mw]  
            r_p_b差第一个维度，因此使用unsqueeze对第一个维度扩充，然后广播机制进行相加
            现：
                context_in：[B, H, L_Q==N, D]
                relative_position_bias：[nH, N=Mh*Mw, N=Mh*Mw]
        '''
        # attn: [B, H, top_u, L_K == N]      index: [B, H, top_u]
        # relative_position_bias：[nH, N=Mh*Mw, N=Mh*Mw] -> [1, H, N, N] -> [B, H, L_K == N, L_K == N]
        # 因为L_K == L_Q, 这样就相当于取出 top Q对应的bias
        from Uformer_Info.options import is_relative_position_bias
        if is_relative_position_bias:
            attn = attn + relative_position_bias.unsqueeze(0).repeat(attn.shape[0], 1, 1, 1)[torch.arange(B)[:, None, None],
                          torch.arange(H)[None, :, None], index, :]
        else:
            attn = attn

        ##########################################################
        """SW-MSA attn加入mask，窗口内不一定都要一起做attention；W-MSA则不加"""
        if SW_mask is not None:  # 说明是SW-MSA，窗口内不一定都要一起做attention
            # SW_mask: [nW, Mh*Mw, Mh*Mw]
            nW = SW_mask.shape[0]  # num_windows
            SW_mask = repeat(SW_mask, 'nW m n -> nW m (n d)', d=ratio)

            # 原：attn:      [batch_size*num_windows, num_heads, N=Mh*Mw, N=Mh*Mw]   * N=Mh*Mw *
            # 现：attn:                               [B, H, top_u, L_K == N]
            # SW_mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]   repeat-> [batch_size, num_windows, num_heads, Mh*Mw, L_K == N == Mh*Mw]
            """之前创建的mask：要做attention的区域值为0(对attention结果无影响)，不做attention的不同区域为-100(之后经过softmax时会趋近0)"""
            num_heads = attn.shape[1]
            SW_mask = SW_mask.unsqueeze(1).unsqueeze(0).repeat(B // nW, 1, num_heads, 1, 1)
            # [B_, H, top_u, L_K == N]       attn.view: [batch_size, num_windows, num_heads, top_u, L_K == N == Mh*Mw]
            # index: [B_, H, top_u]       ---unsqueeze-->       [B_, 1, H, top_u]
            # SW_mask_index [batch_size, num_windows, num_heads, top_u]
            SW_mask_index = index.unsqueeze(1).view(B // nW, nW, num_heads, index.shape[-1])
            attn = attn.view(B // nW, nW, num_heads, attn.shape[-2], attn.shape[-1] * ratio) + SW_mask[
                                                                                               torch.arange(B // nW)[:,
                                                                                               None, None, None],
                                                                                               torch.arange(nW)[None, :,
                                                                                               None, None],
                                                                                               torch.arange(num_heads)[
                                                                                               None, None, :, None],
                                                                                               SW_mask_index, :]

            # attn: [batch_size, num_windows, num_heads, top_u, N]  --->  [batch_size*num_windows, num_heads, top_u, N]
            attn = attn.view(-1, num_heads, attn.shape[-2], attn.shape[-1] * ratio)
            attn = self.softmax(attn)  # 对最后一个维度softmax
        else:
            attn = self.softmax(attn)
        # attn [B_, num_heads, top_u, N]

        # 将那些Top-u中的行的scoreh值更新为 attention值
        # context_in [B, H, L_Q, D]
        # attn: [B, H, M_top, L_K==N]   V:[B, H, L_V, D]
        # matmul [B, H, top_u, L_K] * [B, H, L_V, D] -> [B, H, top_u, D]   事实上L_K == L_V
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] \
            = torch.matmul(attn, V).type_as(context_in)  # 数据类型转化

        # Whether to output attention in encoder(defaults to False)
        if self.output_attention:
            # ones: [B, H, L_V, L_V] 值为 1/L_V   数据类型转换
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    """
    attention(queries, keys, values, attn_mask)
    """

    def forward(self, queries, keys, values, relative_position_bias, SW_mask, attn_mask):
        """
        事实上，输入都为x时，L_Q == L_K == L_V = N
        :param queries: [B, L_Q, H, D]
        :param keys: [B, L_Q, H, D]
        :param values:
        :param attn_mask:
        :return: context [B, L_Q, H, D]  那些Top-u中的行的scoreh值更新为 attention值，其他值为mean(V)
                 attn [B, H, L_V, L_V] 值为 1/L_V
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # [B, H, L_K/Q/V, D]
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # np.ceil计算大于等于该值的最小整数
        """
            U_part 是随机选则Key
            u 是选择 Top_k query with sparisty measurement
        """
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        # U_part 是 U_part和L_K谁小为谁;  u 是 u和L_Q谁小为谁
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # index 找出前top_u的query对应的索引       [B, H, top_u]
        # scores_top： use the 前top_u的query组成的 Q^ to calculate Q_K  [B, H, M_top, L_K]
        """
            正常： Q * K^T -> [B, H, L, L]
            但通过 _prob_QK 计算的 Q^ * K^T [B, H, M_top, L]， 需要进行填充
            是Q_reduce与全部的K做运算
        """
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context
        # _get_initial_context()函数中，先将S所有行都置成 mean(V)
        # [B, H, L_Q, D]
        context = self._get_initial_context(values, L_Q)

        # update the context with selected top_k queries
        # _update_context()函数将那些Top-u中的行的scoreh值更新为 attention值
        # [B, H, L_Q, D]    [B, H, L_V, L_V] 值为 1/L_V
        context, attn = self._update_context(context, values, scores_top, index, L_Q, relative_position_bias, SW_mask,
                                             attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    按照Uformer的代码，该类只需要计算 Q/根号d * K
        * 但是发现Uformer中使用了相对位置偏执，这个时Informer中没有的，Informer计算attention时用到了mean(V),不知道这个影响么
        * SW-MSA中，Q*K^T之后还要加上 mask 保证相邻直接才进行操作，之后才会与 V 相乘
        1. 直接让该类计算attention，偏执和mask作为参数加入，不过还是计算时使用mean(V)，不变  --- 可以保证复杂度不变
            直接加上偏执后就会复杂度提升  让相对位置偏执与全部的Q*K^T相加，而不是Q^*K^T相加
            但是发现没办法全部加，因为代码中会直接得到attention的结果context_in,而Q*K只有部分的结果attn
        所以改进：只对部分结果attn加上 **对应的** Bias

    """

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        """
        AttentionLayer是定义的attention层，会先将输入的embedding分别通过线性映射得到query、key、value。
        还将输入维度 d_model 划分为多头，接着就执行前面定义的attention操作，最后经过一个线性映射得到输出
        :param attention: ProbAttention/self-attention  context [B, L_Q, H, D]     attn [B, H, L_V, L_V] 值为 1/L_V
        :param d_model: 输入的d_model --- 表示Q/K/V的总维度，所以各头维度d_model // n_heads
        :param n_heads: head的个数
        :param d_keys:
        :param d_values:
        :param mix: False
        """
        super(AttentionLayer, self).__init__()

        # 每个head的维度
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = ProbAttention(mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                                             output_attention=False)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, relative_position_bias, SW_mask, attn_mask=None):
        """
        x: [batch_size*num_windows, N=Mh*Mw, total_embed_dim]
        B_, N(Mh*Mw), C = x.shape
        :param queries: x  [B, N, C] [B_=batch_size*num_windows, N=Mh*Mw, total_embed_dim]
        :param keys: x
        :param values: x  用这三个向量产生出每个head的q,k,v
        :param SW_mask: [nW, Mh*Mw, Mh*Mw]
        :param attn_mask:
        :return: [B_, num_heads, N, N]
                 out [B, L_Q, d_model]  attention结果
                 attn [B, H, L_V, L_V] 值为 1/L_V
        """
        # 事实上，输入都为x时，L_Q == L_K == L_V = N
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # [B, L_Q, d_model] -> [B, L_Q, H, D]  D表示每个head的维度
        ####################### 原代码
        # queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        #######################

        ####################### 为了测试参数修改后的代码，使用torchstat测试的时候，linear输入需要两层
        # print('q begin rearange')
        # print(queries.shape)

        queries = rearrange(queries, ' b L c -> (b L) c')
        keys = rearrange(keys, ' b L c -> (b L) c')
        values = rearrange(values, ' b L c -> (b L) c')
        # print('q rearange')
        # print(queries.shape)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        # print('q query_projection')
        # print(queries.shape)

        queries = rearrange(queries, ' (b L) c -> b L c', b=B, L=L).view(B, L, H, -1)
        keys = rearrange(keys, ' (b L) c -> b L c', b=B, L=S).view(B, S, H, -1)
        values = rearrange(values, ' (b L) c -> b L c', b=B, L=S).view(B, S, H, -1)
        # print('q last rearange')
        # print(queries.shape)
        #######################

        # context [B, L_Q, H, D]那些Top-u中的行的scoreh值更新为 attention值，其他值为mean(V)
        # attn [B, H, L_V, L_V] 值为 1/L_V
        out, attn = self.inner_attention(
            queries,  # [B, N, H, D]
            keys,
            values,
            relative_position_bias,
            SW_mask,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()

        # [B, L_Q, d_model]
        out = out.view(B, L, -1)

        ####################### 原代码
        # return self.out_projection(out), attn
        #######################

        ####################### 为了测试参数修改后的代码，使用torchstat测试的时候，linear输入需要两层
        out = rearrange(out, ' b L c -> (b L) c')

        out_projection = self.out_projection(out)

        out_projection = rearrange(out_projection, ' (b L) c -> b L c', b=B, L=L)
        #######################

        return out_projection, attn
