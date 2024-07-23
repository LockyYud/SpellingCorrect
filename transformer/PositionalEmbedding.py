import torch
import torch.nn as nn
import numpy as np


def positional_encoding(length, depth):
    depth = depth // 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return torch.tensor(pos_encoding, dtype=torch.float32)


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, positional=True):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional = positional
        if positional:
            self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def forward(self, x):
        length = x.size(1)
        # x.size() return torch.Size([3200, 10])
        # if torch.max(x) >= self.embedding.num_embeddings:
        #     raise ValueError("Input indices are out of range for the embedding layer.")
        x = self.embedding(x)
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(x.device)
        if self.positional:
            x = x + self.pos_encoding[:length, :].unsqueeze(0).to(x.device)
        return x
