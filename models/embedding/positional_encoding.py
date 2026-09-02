import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        encoding = torch.zeros(max_len, d_model, device=device)

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        scale = 10000 ** (_2i / d_model)
        encoding[:, 0::2] = torch.sin(pos / scale)
        encoding[:, 1::2] = torch.cos(pos / scale[: d_model // 2])
        self.register_buffer("encoding", encoding, persistent=False)
        # compute positional encoding to consider positional information of words

    def forward(self, x, offset=0):
        # self.encoding
        # [max_len = 512, d_model = 512]

        seq_len = x.size(1)
        # [batch_size = 128, seq_len = 30]

        end = int(offset) + seq_len
        if offset < 0 or end > self.encoding.shape[0]:
            raise ValueError("Requested positions exceed the configured max_len.")
        return self.encoding[offset:end, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
