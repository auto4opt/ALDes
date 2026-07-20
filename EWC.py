"""Diagonal-Fisher elastic weight consolidation for continual ALDes."""

from __future__ import annotations

import torch
from torch import nn


class EWC:
    def __init__(self, model: nn.Module):
        self._means = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self._precision_matrices = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update_diag_fisher(self, model: nn.Module) -> None:
        """Accumulate squared policy gradients for one sampled batch."""

        for name, parameter in model.named_parameters():
            if name in self._precision_matrices and parameter.grad is not None:
                self._precision_matrices[name] += parameter.grad.detach().square()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.zeros((), device=next(model.parameters()).device)
        for name, parameter in model.named_parameters():
            if name in self._precision_matrices:
                loss = (
                    loss
                    + (
                        self._precision_matrices[name]
                        * (parameter - self._means[name]).square()
                    ).sum()
                )
        return loss
