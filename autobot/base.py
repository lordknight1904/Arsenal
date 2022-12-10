import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self,
        emb_dim, attn_dim, v_dim,
        num_heads,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.v_dim = v_dim
        self.num_heads = num_heads

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

    def build_query(self): return nn.Linear(self.emb_dim, self.attn_dim)

    def build_key(self): return nn.Linear(self.emb_dim, self.attn_dim)

    def build_value(self): return nn.Linear(self.emb_dim, self.v_dim)

    def build_linear(self): return nn.Linear(self.emb_dim, self.v_dim)
        
    def _headify(self, x):
        return x

    def _straight(self, x):
        return x

    def _calculate_attn(self, q, k):
        return torch.matmul(q, k.transpose(-1,-2))

    def forward(self, s, t):
        q = self.query(self._headify(t))
        k = self.key(self._headify(s))
        v = self.value(self._headify(s))

        attn = self._calculate_attn(q, k)

        o = self.linear(torch.matmul(attn, v))

        return o


class RelativeMultiHeadAttention(MultiHeadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
