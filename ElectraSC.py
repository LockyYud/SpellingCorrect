import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer
from SelfAttention import SelfAttention
from PositionalEmbedding import PositionalEmbedding


class ElectraSC(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dff,
        vocab_size,
        num_layer,
        device,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.embedding_layer = PositionalEmbedding(
            vocab_size=vocab_size, d_model=embed_dim
        ).to(device)
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
            nn.Linear(embed_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 2, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classifier_layer(x)
