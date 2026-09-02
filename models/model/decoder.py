"""Autoregressive decoder used by the standalone ALDes training script."""

from __future__ import annotations

import math

import numpy as np
import torch
from autooptlib.aldes.vocabulary import VOCABULARY_SIZE, allowed_next_tokens
from torch import nn

from aldes_setting import begin_index, end_index
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
        if dec_voc_size != VOCABULARY_SIZE:
            raise ValueError(f"ALDes requires a {VOCABULARY_SIZE}-token vocabulary.")
        if max_len < 2 or n_layers <= 0:
            raise ValueError(
                "max_len must be at least 2 and n_layers must be positive."
            )
        if ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive.")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1).")
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
        if trg.ndim != 2 or trg.shape[1] == 0:
            raise ValueError("Target tokens must have shape (batch, length>=1).")
        if trg.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Target tokens must use an integer tensor dtype.")
        if not bool((trg[:, 0] == begin_index).all()):
            raise ValueError("Every target prefix must start with the begin token.")
        if bool(((trg < 0) | (trg >= VOCABULARY_SIZE)).any()):
            raise ValueError("Target prefix contains an unknown token.")
        for row in trg:
            prefix = [int(row[0])]
            for token in row[1:].tolist():
                if not allowed_next_tokens(prefix)[int(token)]:
                    raise ValueError("Target prefix is not grammar-valid.")
                prefix.append(int(token))
        if bool((trg == end_index).any()):
            raise ValueError("Target prefix must not already contain the end token.")
        if action is not None:
            raw_action = torch.as_tensor(action, device=trg.device)
            if raw_action.is_complex() or (
                raw_action.is_floating_point()
                and not torch.equal(raw_action, raw_action.round())
            ):
                raise ValueError("Replay actions must contain integer token values.")
            action = raw_action.to(dtype=torch.long)
            if action.ndim != 2 or action.shape[0] != trg.shape[0]:
                raise ValueError("Replay action batch must match the target batch.")
            if action.shape[1] <= trg.shape[1] or not torch.equal(
                action[:, : trg.shape[1]], trg
            ):
                raise ValueError("Replay actions must extend the target prefix.")
            if bool(((action < 0) | (action >= VOCABULARY_SIZE)).any()):
                raise ValueError("Replay action contains an unknown token.")
        if not self.condition_on_features and enc_src is not None:
            raise ValueError("Single-problem ALDes mode does not accept features.")
        action_p = []
        action_log_p = []
        ppo_index = trg.shape[1]

        while True:
            if self.temp <= 0:
                raise ValueError("Decoder temperature must be positive.")
            if trg.shape[1] >= self.max_len:
                raise RuntimeError("ALDes generation reached max_len before end.")

            position_offset = 1 if self.condition_on_features else 0
            decoder_input = self.emb(trg, position_offset=position_offset)
            if self.condition_on_features:
                if enc_src is None:
                    raise ValueError("Continual ALDes mode requires problem features.")
                feature = enc_src
                if feature.ndim == 3:
                    feature = feature.mean(dim=1)
                if feature.ndim != 2 or feature.shape[-1] != decoder_input.shape[-1]:
                    raise ValueError(
                        "Problem features must have shape (batch, d_model) or "
                        "(batch, samples, d_model)."
                    )
                if feature.shape[0] == 1 and trg.shape[0] > 1:
                    feature = feature.expand(trg.shape[0], -1)
                if feature.shape[0] != trg.shape[0]:
                    raise ValueError("Feature batch must match the target batch.")
                feature_token = self.feature_projection(feature).unsqueeze(1)
                decoder_input = torch.cat((feature_token, decoder_input), dim=1)

            sequence_length = decoder_input.shape[1]
            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=decoder_input.device,
                )
            )
            for layer in self.layers:
                decoder_input = layer(decoder_input, trg_mask=causal_mask)
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

            legal = mask.squeeze(1).gather(1, action_index)
            if not bool(legal.all()):
                raise ValueError("Replay action contains a grammar-invalid token.")

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
                if action is not None and ppo_index != action.shape[1]:
                    raise ValueError("Replay action contains tokens after termination.")
                break

        return trg, torch.stack(action_p, 1), torch.stack(action_log_p, 1)

    @staticmethod
    def check_pointer(cur_alg):
        """Compatibility no-op; AutoOptLib safely interprets fork targets."""

        return cur_alg

    def get_mask(self, cur_alg):
        rows = [allowed_next_tokens(row.detach().cpu().numpy()) for row in cur_alg]
        return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=cur_alg.device)
