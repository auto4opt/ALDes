from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module):

        self.model = model

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = None #self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        #self.model.eval()
        for n, p in self.model.named_parameters():
            if p.grad != None:
                precision_matrices[n].data += p.grad.data ** 2
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def update_diag_fisher(self,model):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        for n, p in model.named_parameters():
            if p.grad != None:
                precision_matrices[n].data += p.grad.data ** 2
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        if self._precision_matrices is None:
            self._precision_matrices = precision_matrices
        else:
            for key in precision_matrices:
                self._precision_matrices[key] +=precision_matrices[key]
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss*100