from torch import nn

from models.model.decoder import Decoder


class Transformer(nn.Module):
    """Decoder-only Transformer policy used by ALDes."""

    def __init__(
        self,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
        condition_on_features=False,
    ):
        super().__init__()
        self.device = device
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device,
            condition_on_features=condition_on_features,
        )

    def forward(
        self,
        features,
        target,
        attention_positions,
        action=None,
        reference=False,
    ):
        # ``attention_positions`` is retained in the public call signature for
        # compatibility with the original ALDes training loop. The decoder
        # derives positional information internally.
        del attention_positions
        return self.decoder(
            target,
            features,
            None,
            None,
            None,
            action,
            reference,
        )
