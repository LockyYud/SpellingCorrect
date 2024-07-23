import torch
import torch.nn as nn

from custom_layer.EncoderCharacter import EncoderCharacter
from custom_layer.EncoderBERTpretrain import EncoderBERTpretrain
from transformer.EncoderLayer import EncoderLayer
from transformer.PositionalEmbedding import PositionalEmbedding

# input: word_feature: (batch_size x sen_len x 768), chars_not_embedded: (batch_size x sen_len x word_len) -
# output: (batch_size x sen_len)


class ModelSC(nn.Module):
    def __init__(
        self,
        character_level_d_model,
        word_level_d_model,
        num_heads_char_encoder,
        num_layers_char_encoder,
        num_heads_word_encoder,
        num_layers_word_encoder,
        dff,
        character_vocab_size,
        word_vocab_size,
        device,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.encoder_character = EncoderCharacter(
            num_heads=num_heads_char_encoder,
            num_layers=num_layers_char_encoder,
            vocab_size=character_vocab_size,
            device=device,
            dff=dff,
            dropout_rate=dropout_rate,
            embed_dim=character_level_d_model,
        ).to(device)
        self.embedding_word = PositionalEmbedding(
            vocab_size=word_vocab_size, d_model=word_level_d_model
        )
        self.encoder_word = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=word_level_d_model,
                    num_heads=num_heads_word_encoder,
                    dropout=dropout_rate,
                    dff=dff,
                ).to(device)
                for _ in range(num_layers_word_encoder)
            ]
        ).to(device=device)
        self.linear = nn.Sequential(
            nn.Linear(
                character_level_d_model + word_level_d_model,
                character_level_d_model + word_level_d_model,
            ),
            nn.LayerNorm(
                character_level_d_model + word_level_d_model,
                eps=1e-05,
                elementwise_affine=True,
            ),
            nn.Linear(character_level_d_model + word_level_d_model, word_vocab_size),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, *inputs):
        # Input
        char_input_ids, word_input_ids, masked_positions = inputs

        # Encode character
        key_mask_padding = word_input_ids != 1
        output_char_encoder = self.encoder_character(char_input_ids, key_mask_padding)
        output_char_encoder = output_char_encoder.view(
            word_input_ids.size(0), word_input_ids.size(1), -1
        )
        # Encode word
        output_word_encode = self.embedding_word(word_input_ids)
        # return 1
        for layer in self.encoder_word:
            output_word_encode = layer(output_word_encode, word_input_ids == 1)
        # Combine encode
        output = torch.cat((output_word_encode, output_char_encoder), -1)
        attn_mask = word_input_ids != 1
        attn_mask = attn_mask[:, None, None, :]
        output = output[masked_positions]
        output = self.linear(output)
        return output
