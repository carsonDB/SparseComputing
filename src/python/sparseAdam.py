from typing import Optional, List, Tuple, Union
import torch
from torch import Tensor
from torch.optim import _functional as F

from .sparseTypes import SparseTensor, SparseParameter


class SparseAdam(torch.optim.Optimizer):
    """Inherit from: 
        SparseAdam: https://pytorch.org/docs/stable/_modules/torch/optim/sparse_adam.html#SparseAdam
        F.sparse_adam: https://github.com/pytorch/pytorch/blob/master/torch/optim/_functional.py
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        params = list(params)

        sparse_params = []
        for index, param in enumerate(params):
            if isinstance(param, dict):
                for d_index, d_param in enumerate(param.get("params", [])):
                    assert isinstance(d_param, (SparseTensor, SparseParameter, Tensor)), \
                        "only allow Tensor / SparseTensor"
            else:
                assert isinstance(param, (SparseTensor, SparseParameter, Tensor)), \
                        "only allow Tensor / SparseTensor"

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            eps = group['eps']
            lr = group['lr']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # p maybe sparseTensor paramters, should use p.data directly.
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            F.sparse_adam(params_with_grad,
                          grads,
                          exp_avgs,
                          exp_avg_sqs,
                          state_steps,
                          beta1=beta1,
                          beta2=beta2,
                          lr=group['lr'],
                          eps=group['eps'])

        return loss
