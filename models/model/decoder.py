"""Autoregressive decoder used by the standalone ALDes training script."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from aldes_setting import end_index
from autooptlib.aldes.vocabulary import allowed_next_tokens
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
        condition_on_features=False,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(d_model, dec_voc_size)
        self.device = device
        self.max_len = max_len
        self.condition_on_features = condition_on_features
        self.feature_projection = (
            nn.Linear(d_model, d_model) if condition_on_features else None
        )
        self.temp = 1.0

    def forward(
        self,
        trg,
        enc_src,
        action_emb,
        trg_mask,
        src_mask,
        action=None,
        inference=False,
    ):
        del action_emb, trg_mask, src_mask
        action_p = []
        action_log_p = []
        ppo_index = 1

        while True:
            if trg.shape[1] >= self.max_len:
                raise RuntimeError("ALDes generation reached max_len before end.")

            decoder_input = self.emb(trg)
            if self.condition_on_features:
                if enc_src is None:
                    raise ValueError("Continual ALDes mode requires problem features.")
                feature = enc_src
                if feature.ndim == 3:
                    feature = feature.mean(dim=1)
                if feature.ndim != 2 or feature.shape[-1] != decoder_input.shape[-1]:
                    raise ValueError(
                        "Problem features must have shape (batch, 32) or "
                        "(batch, samples, 32)."
                    )
                decoder_input = decoder_input + self.feature_projection(
                    feature
                ).unsqueeze(1)

            for layer in self.layers:
                decoder_input = layer(decoder_input, trg_mask=None)
            output = self.linear(decoder_input[:, -1:, :])
            mask = self.get_mask(trg).unsqueeze(1)
            output = output.masked_fill(~mask, -math.inf)
            log_probabilities = torch.log_softmax(output / self.temp, dim=-1)
            probabilities = log_probabilities.exp()

            if action is None:
                if inference:
                    action_index = probabilities.argmax(dim=-1)
                else:
                    action_index = probabilities.squeeze(1).multinomial(1)
            else:
                if ppo_index >= action.shape[1]:
                    raise ValueError("Replay action ends before the end token.")
                action_index = action[:, ppo_index].unsqueeze(1)
                ppo_index += 1

            probability = probabilities.gather(2, action_index.unsqueeze(-1)).squeeze(
                -1
            )
            log_probability = log_probabilities.gather(
                2, action_index.unsqueeze(-1)
            ).squeeze(-1)
            trg = torch.cat([trg, action_index], dim=1)
            action_p.append(probability)
            action_log_p.append(log_probability)
            if torch.all(action_index == end_index):
                break

        return trg, torch.stack(action_p, 1), torch.stack(action_log_p, 1)

    @staticmethod
    def check_pointer(cur_alg):
        """Compatibility no-op; AutoOptLib safely interprets fork targets."""

        return cur_alg

    def get_mask(self, cur_alg):
        rows = [allowed_next_tokens(row.detach().cpu().numpy()) for row in cur_alg]
        return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=self.device)
