"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """Compute scaled dot-product attention."""

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # Inputs have shape [batch, head, sequence length, head dimension].
        d_tensor = k.size(-1)

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v

        return v, score
