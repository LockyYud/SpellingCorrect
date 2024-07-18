import torch
import torch.nn as nn
from transformer.EncoderLayer import EncoderLayer
from transformer.SelfAttention import SelfAttention
from transformer.PositionalEmbedding import PositionalEmbedding


class DetectModel(nn.Module):
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
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    dff=dff,
                ).to(device)
                for _ in range(num_layer)
            ]
        )
        self.classifier_layer = nn.Sequential(
            nn.Linear(embed_dim, 2, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        key_mask_padding = x == 1
        x = self.embedding_layer(x)
        for layer in self.encoder_layers:
            x = layer(x, key_mask_padding)
        return self.classifier_layer(x)
