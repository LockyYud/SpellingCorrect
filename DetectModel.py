import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer
from SelfAttention import SelfAttention
from PositionalEmbedding import PositionalEmbedding


class DetectModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dff,
        vocab_size,
        num_layer,
        device,
        phobert_embed,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.embedding_layer = phobert_embed
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    dff=dff,
                ).to(device)
                for i in range(num_layer)
            ]
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(embed_dim, dff, bias=True),
            nn.Linear(dff, 2, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        key_maks_padding = x == 1
        x = self.embedding_layer(x)
        for layer in self.encoder_layers:
            x = layer(x, key_maks_padding)
        return self.classifier_layer(x)
