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

    def update_diag_fisher(
        self,
        model: nn.Module,
        log_likelihoods: torch.Tensor | None = None,
    ) -> None:
        """Accumulate an empirical diagonal Fisher estimate.

        When per-sample log likelihoods are supplied, Fisher entries are the
        mean of the squared score gradients.  The optional gradient fallback
        is retained for compatibility with callers that already performed a
        backward pass.
        """

        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name in self._precision_matrices
        ]
        if log_likelihoods is None:
            for name, parameter in named_parameters:
                if parameter.grad is not None:
                    self._precision_matrices[name] += parameter.grad.detach().square()
            return

        values = log_likelihoods.reshape(-1)
        if values.numel() == 0:
            raise ValueError("At least one policy log likelihood is required.")
        if not values.requires_grad:
            raise ValueError("Policy log likelihoods must retain their gradients.")

        parameters = [parameter for _, parameter in named_parameters]
        weight = 1.0 / values.numel()
        for index, value in enumerate(values):
            gradients = torch.autograd.grad(
                value,
                parameters,
                retain_graph=index + 1 < values.numel(),
                allow_unused=True,
            )
            for (name, _), gradient in zip(named_parameters, gradients):
                if gradient is not None:
                    self._precision_matrices[name] += (
                        gradient.detach().square() * weight
                    )

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
        # Equation (14) uses lambda / 2 times the Fisher-weighted distance.
        return 0.5 * loss
