import torch
from torch import nn
from transformer.PositionalEmbedding import PositionalEmbedding
from transformer.EncoderLayer import EncoderLayer


class EncoderCharacter(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        embed_dim,
        dff,
        vocab_size,
        device,
        dropout_rate,
    ) -> None:
        super().__init__()
        self.embedding_layer = PositionalEmbedding(
            vocab_size=vocab_size, d_model=embed_dim
        ).to(device=device)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    dff=dff,
                ).to(device)
                for i in range(num_layers)
            ]
        )
        self.device = device
        self.encoder_word_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    dff=dff,
                ).to(device)
                for i in range(num_layers)
            ]
        )

    def forward(self, *inputs):
        # input_ids (batch_size x sentence_len x word_len)
        input_ids, word_padding = inputs
        batch_size, sentence_len, word_len = input_ids.size()
        # input_ids ((batch_size * sentence_len) x word_len)
        input_ids = input_ids.view(batch_size * sentence_len, word_len)
        word_padding = word_padding.view(batch_size * sentence_len)
        input_embed = input_ids[word_padding]
        output = self.embedding_layer(input_embed)
        key_mask_padding = input_embed == 1
        for layer in self.encoder_layers:
            output = layer(output, key_mask_padding)
        output_padding = torch.zeros(
            [batch_size * sentence_len, word_len, output.size(-1)]
        ).to(device=self.device)
        output_padding[word_padding] = output
        output_padding.reshape(
            batch_size, sentence_len, word_len, output_padding.size(-1)
        )
        output_padding = output_padding.mean(dim=1).view(
            batch_size, sentence_len, output_padding.size(-1)
        )
        key_mask_word_padding = torch.ones([batch_size, sentence_len]).to(
            device=self.device
        )
        key_mask_word_padding[word_padding.view(batch_size, sentence_len)] = False
        for layer in self.encoder_word_layers:
            output_padding = layer(output_padding, key_mask_word_padding)
        return output_padding
