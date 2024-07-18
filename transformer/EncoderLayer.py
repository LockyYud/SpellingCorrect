import torch.nn as nn
import torch
from transformer.FeedForward import FeedForward
from transformer.SelfAttention import SelfAttention


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, dff):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fn = FeedForward(d_model=embed_dim, dff=dff)

    def forward(self, *input):
        x = self.self_attention(*input)
        x = self.fn(x)
        return x
