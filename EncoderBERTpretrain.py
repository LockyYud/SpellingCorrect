import torch
from torch import nn
from PositionalEmbedding import PositionalEmbedding
from EncoderLayer import EncoderLayer


class EncoderBERTpretrain(nn.Module):
    def __init__(self, embedding_layer, encoder_layers, device) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer.to(device)
        self.encoder_layers: nn.ModuleList = encoder_layers.to(device)

    def forward(self, input_phobert_ids, embeded_char):
        input_ids = input_phobert_ids
        attn_mask = input_ids != 1
        attn_mask = attn_mask[:, None, None, :]
        embeded_phobert = self.embedding_layer(input_ids)
        output = torch.cat((embeded_phobert, embeded_char), dim=-1)
        for layer in self.encoder_layers:
            output = layer(output, attention_mask=attn_mask)[0]
        return output
