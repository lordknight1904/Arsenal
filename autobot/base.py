import math

import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, emb_dim, drop_prob=0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, emb_dim*4)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(emb_dim*4, emb_dim)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return self.drop(x)


class MultiHeadAttention(nn.Module):

    def __init__(self,
        attn_dim, v_dim,
        num_heads,
        drop_prob=0.1
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.v_dim = v_dim
        self.num_heads = num_heads

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

        self.attn_drop = nn.Dropout(drop_prob)
        self.v_drop = nn.Dropout(drop_prob)

    def build_query(self): return nn.Linear(self.v_dim, self.attn_dim)

    def build_key(self): return nn.Linear(self.v_dim, self.attn_dim)

    def build_value(self): return nn.Linear(self.v_dim, self.v_dim)

    def build_linear(self): return nn.Linear(self.v_dim, self.v_dim)

    def _straight(self, x):
        return x

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, int(self.attn_dim / self.num_heads))
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def _logits(self, q, k, mask):
        logits = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(int(self.attn_dim // self.num_heads))
        if mask:
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits

    def _attn(self, logits):
        return torch.softmax(logits, dim=-1)

    def forward(self, x, y, mask=None):
        b, s, *_ = x.shape
        _, t, *_ = y.shape

        query = self.transpose_for_scores(self.query(y))
        key = self.transpose_for_scores(self.key(x))
        value = self.transpose_for_scores(self.value(x))

        logits = self._logits(query, key, mask)
        attn = self._attn(logits)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, value).contiguous().permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.v_dim,))

        o = self.linear(context)
        o = self.v_drop(o)

        return o


class EncoderBlock(nn.Module):

    def __init__(self,
        # mha_cls,
        emb_dim, attn_dim,
        num_heads,
        drop_prob=0.1
    ):
        super().__init__()

        self.mha = MultiHeadAttention(attn_dim, emb_dim, num_heads, drop_prob)
        self.mha_norm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.ff = FeedForward(emb_dim)
        self.ff_norm = nn.LayerNorm(emb_dim, eps=1e-12)

    def forward(self, x, mask=None):
        x_ = x
        x = self.mha_norm(x)
        x = self.mha(x, x, mask)
        x_ = x + x_

        x = self.ff_norm(x_)
        x = self.ff(x)
        x = x + x_

        return x
