import torch
from torch import nn
from PositionalEmbedding import PositionalEmbedding
from EncoderLayer import EncoderLayer


class EncoderBERTpretrain(nn.Module):
    def __init__(self, embedding_layer, encoder_layers, device) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer.to(device)
        self.encoder_layers: nn.ModuleList = encoder_layers.to(device)

    def forward(self, inputs):
        input_ids = inputs
        attn_mask = input_ids != 1
        attn_mask = attn_mask[:, None, None, :]
        output = self.embedding_layer(input_ids)
        for layer in self.encoder_layers:
            output = layer(output, attention_mask=attn_mask)[0]
        return output
