import torch
import torch.nn as nn

from EncoderCharacter import EncoderCharacter
from EncoderBERTpretrain import EncoderBERTpretrain

# input: word_feature: (batch_size x sen_len x 768), chars_not_embedded: (batch_size x sen_len x word_len) -
# output: (batch_size x sen_len)


class ModelSC(nn.Module):
    def __init__(
        self,
        character_level_d_model,
        word_level_d_model,
        num_heads,
        dff,
        character_vocab_size,
        word_vocab_size,
        num_layers,
        device,
        bert_embedding_layer,
        bert_encoder_layers,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.EncoderCharacter = EncoderCharacter(
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=character_vocab_size,
            device=device,
            dff=dff,
            dropout_rate=dropout_rate,
            embed_dim=character_level_d_model,
        ).to(device)
        self.EncoderBERTpretrain = EncoderBERTpretrain(
            embedding_layer=bert_embedding_layer,
            encoder_layers=bert_encoder_layers,
            device=device,
        ).to(device)
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
        )

    def forward(self, *inputs):
        char_input_ids, bert_input_ids, masked_positions = inputs
        key_mask_padding = bert_input_ids != 1
        output_charEncoder = self.EncoderCharacter(char_input_ids, key_mask_padding)
        output_charEncoder = output_charEncoder.view(
            bert_input_ids.size(0), bert_input_ids.size(1), -1
        )
        output = self.EncoderBERTpretrain(bert_input_ids, output_charEncoder)
        output = output[masked_positions]
        output = self.linear(output)
        return output
