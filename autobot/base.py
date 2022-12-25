import torch
from torch import nn


# class APE(nn.Module):

#     def __init__(self, d_model, max_seq_len=80):
#         super().__init__()
#         self.d_model = d_model
#         pe = torch.zeros(max_seq_len, d_model)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = \
#                 math.sin(pos / (10000 ** ((2 * i)/d_model)))
#                 pe[pos, i + 1] = \
#                 math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:,:seq_len]* math.sqrt(self.d_model)


class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, emb_dim*4)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(emb_dim*4, emb_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


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

    def build_query(self): return nn.Linear(self.emb_dim*self.num_heads, self.attn_dim*self.num_heads)

    def build_key(self): return nn.Linear(self.emb_dim*self.num_heads, self.attn_dim*self.num_heads)

    def build_value(self): return nn.Linear(self.emb_dim*self.num_heads, self.v_dim*self.num_heads)

    def build_linear(self): return nn.Linear(self.emb_dim*self.num_heads, self.v_dim*self.num_heads)
        
    # def _headify(self, x):
    #     b, l, *_ = x.shape
    #     return x.view(b, l, self.num_heads, self.d_head).transpose(1,2)

    def _straight(self, x):
        return x

    def _calculate_logits(self, q, k, mask):
        logits = torch.matmul(q, k.transpose(-1,-2)) / torch.sqrt(self.attn_dim)
        if mask:
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits

    def _calculate_attn(self, logits):
        return torch.softmax(logits, dim=-1)

    def forward(self, x, y, mask=None):
        b, s, *_ = x.shape
        _, t, *_ = y.shape

        query = self.query(y).view(b, t, self.num_heads, self.attn_dim).transpose(1, 2)
        key = self.key(x).view(b, s, self.num_heads, self.attn_dim).transpose(1, 2)
        value = self.value(x).view(b, s, self.num_heads, self.v_dim).transpose(1, 2)

        logits = self._calculate_attn(query, key, mask)
        attn = self._calculate_attn(logits)

        o = self.linear(
            torch.matmul(attn, value).contiguous().view(b, -1, self.v_dim)
        )

        return o


class RelativeMultiHeadAttention(MultiHeadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.rel_emb = 

    def forward(self, x, H, W):

        return x


class EncoderBlock(nn.Module):

    def __init__(self,
        emb_dim, attn_dim, v_dim,
        num_heads,
        drop_prob=0.1
    ):
        super().__init__()

        self.mha = MultiHeadAttention(emb_dim, attn_dim, v_dim, num_heads)
        self.drop = nn.Dropout(drop_prob)
        self.mha_norm = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)
        self.ff_norm = nn.LayerNorm(emb_dim)


class RelativeTransformer(nn.Module):  # Pre-norm

    def __init__(self,
        emb_dim, attn_dim, v_dim,
        num_heads,
        drop_prob=0.1
    ):
        super().__init__()

        self.mha = MultiHeadAttention(emb_dim, attn_dim, v_dim, num_heads)
        self.drop = nn.Dropout(drop_prob)
        self.mha_norm = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)
        self.ff_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, H, W):
        x_ = self.mha_norm(x)
        x_, score = self.mha(x_, x_, x_)
        x = x + self.drop(x_)
            
        x_ = self.ff_norm(x)
        x = x + self.ff(x_)

        return x