import torch
from torch import nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_softmoe(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 多头注意力网络
        self.Transformer_a = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, cross_modality='atv', mask_modality=None, mask=None):
        # x: [B, s, C]
        B, s, C = x.shape
        if cross_modality == 'a':
            x_a_mlp = self.Transformer_a(x, mask_modality, mask)
            return x_a_mlp
        if cross_modality == 't':
            x_t_mlp = self.Transformer_t(x, mask_modality, mask)
            return x_t_mlp
        if cross_modality == 'v':
            x_v_mlp = self.Transformer_v(x, mask_modality, mask)
            return x_v_mlp


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5  # 用于缩放查询（Q），防止点积注意力得分过大，影响梯度稳定性
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)  # 用于注意力权重的 dropout，防止过拟合
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )

    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape

        # 调整维度顺序为 [B, num_heads, seq_len, head_dim]，便于并行计算多头注意力。
        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale  # 缩放查询，防止点积过大。
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        if mask is not None:
            mask = mask.bool()
            # 按模态（mask_modality）分割掩码，并根据 mask_modality 选择对应的掩码
            mask = {'a': mask[:, :seq_len], 't': mask[:, seq_len:2 * seq_len], 'v': mask[:, 2 * seq_len:3 * seq_len]}
            mask = mask[mask_modality]
            # 使用 masked_fill 将不需要关注的位置设置为 -inf，以便在 softmax 后这些位置的权重接近于零
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            # 处理可能出现的 NaN 值，将其替换为零
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)

        return x_out


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.drop = drop

        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio, )
                for i in range(depth)
            ]
        )

    def forward(self, x, first_stage, mask=None, modality=None):
        if first_stage:
            for layer_idx, block in enumerate(self.blocks):
                # 有残差连接，cross_modality和mask_modality相同，只编码一种模态
                x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
            return x
        else:
            x_cross_a, x_cross_t, x_cross_v = torch.clone(x), torch.clone(x), torch.clone(x)
            for layer_idx, block in enumerate(self.blocks):
                # 残差连接，编码器内部同时编码三种模态，并将编码结果进行拼接作为输出
                x_cross_a = x_cross_a + block(x_cross_a, cross_modality='a', mask_modality=modality, mask=mask)
                x_cross_t = x_cross_t + block(x_cross_t, cross_modality='t', mask_modality=modality, mask=mask)
                x_cross_v = x_cross_v + block(x_cross_v, cross_modality='v', mask_modality=modality, mask=mask)
            return torch.cat([x_cross_a, x_cross_t, x_cross_v], dim=-1)


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )

    def forward(self, query, key, value, mask_modality_a=None, mask_modality_b=None, mask=None):
        """
        参数:
            query: [B, seq_len_q, C]
            key: [B, seq_len_k, C]
            value: [B, seq_len_v, C]
            mask_modality: 模态类型，用于选择不同的掩码
            mask: [B, seq_len_k * num_modalities]，用于掩码操作
        返回:
            x_out: [B, seq_len_q, C]
        """
        B, seq_len_q, C = query.shape
        _, seq_len_k, _ = key.shape
        _, seq_len_v, _ = value.shape
        assert seq_len_q == seq_len_k == seq_len_v, "Query, Key 和 Value 的序列长度必须相同"
        assert C % self.num_heads == 0, "特征维度必须能被注意力头数整除"

        # 生成 Q, K, V
        q = self.q(query).reshape(B, seq_len_q, self.num_heads, -1).permute(0, 2, 1,
                                                                            3)  # [B, heads, seq_len_q, head_dim]
        k = self.k(key).reshape(B, seq_len_k, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, heads, seq_len_k, head_dim]
        v = self.v(value).reshape(B, seq_len_v, self.num_heads, -1).permute(0, 2, 1,
                                                                            3)  # [B, heads, seq_len_v, head_dim]
        # 缩放 Q
        q = q * self.scale

        # 计算注意力得分
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, heads, seq_len_q, seq_len_k]

        if mask is not None:
            mask = mask.bool()
            # 假设 mask 的形状为 [B, seq_len_k * num_modalities]
            # 根据 mask_modality 选择对应的部分
            num_modalities = 3  # 示例：假设有 3 种模态
            assert mask.shape[1] == seq_len_k * num_modalities, "掩码长度与模态数不匹配"
            modality_masks = {
                'a': mask[:, :seq_len_k],
                't': mask[:, seq_len_k:2 * seq_len_k],
                'v': mask[:, 2 * seq_len_k:3 * seq_len_k],
            }
            selected_mask_a = modality_masks.get(mask_modality_a, None)
            selected_mask_b = modality_masks.get(mask_modality_b, None)
            combined_mask = selected_mask_a | selected_mask_b  # 逻辑或
            attn = self.attn_drop(
                attn.masked_fill(~combined_mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(q))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        # 计算加权值
        x = torch.matmul(attn, v)  # [B, heads, seq_len_q, head_dim]
        x = x.transpose(1, 2).reshape(B, seq_len_q, C)  # [B, seq_len_q, C]

        # 线性变换和 dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        # 前馈网络和残差连接
        x = x + self.mlp(x)

        return x


class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.drop = drop

        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio, )
                for i in range(depth)
            ]
        )

    def forward(self, x, mask=None, modality=None):
        for layer_idx, block in enumerate(self.blocks):
            # 有残差连接，cross_modality和mask_modality相同，只编码一种模态
            x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
        return x
