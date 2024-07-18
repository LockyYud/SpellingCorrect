from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        # self.mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,dropout=dropout)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(normalized_shape=embed_dim)

    # @torch.jit.unused(
    def forward(self, *inputs):
        # x (batch_size x seq_length x embed_dim), mask (batch_size . numhead x seq_length x seq_length)
        x, key_padding_mask = inputs
        (attn_output, attn_output_weight) = self.mha(
            query=x, value=x, key=x, key_padding_mask=key_padding_mask
        )
        x = x + attn_output
        x = self.layernorm(x)
        return x
