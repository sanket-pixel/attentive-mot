import torch
import torch.nn.functional as f
from torch import nn


def scaled_dot_product_attention(query, key, value):
    attention = torch.bmm(query, key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(attention / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query, key, value):
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query, key, value):
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

def feed_forward(dim_input,dim_feedforward):
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual(nn.Module):
    def __init__(self,sublayer, dimension, dropout):
        super(Residual, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        return self.norm(tensors[-1]+self.dropout(self.sublayer(*tensors)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model = 64,
                 num_heads = 8,
                 dim_feedforward = 128,
                 dropout = 0.1,
                 ):
        super(TransformerEncoderLayer, self).__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension = dim_model,
            dropout = dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout = dropout,
        )
    def forward(self, src):
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers = 6,
                 dim_model = 64,
                 num_heads = 8,
                 dim_feedforward = 128,
                 dropout = 0.1,
                 ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,src):
        for layer in self.layers:
            src = layer(src)

        return src



#head_params
num_heads = 8
dim_in = 64
dim_k = dim_v = dim_in // num_heads

#data params
batch_size = 16
sequence_length = 10
num_features = 64

query = torch.rand(batch_size, sequence_length,num_features)
key = torch.rand(batch_size, sequence_length,num_features)
value = torch.rand(batch_size, sequence_length,num_features)

transformer_encoder = TransformerEncoder()
transformer_encoder(value)

