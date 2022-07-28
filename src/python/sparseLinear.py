from typing import Optional, List, Tuple, Union
from torch import nn, Tensor
import timeit
from torch.autograd import Function
import torch

import sparseOps_cpp as spCpp
from .sparseTypes import SparseTensor, SparseParameter


class SparseLinearFunction(Function):
    @staticmethod
    def forward(ctx, input: SparseTensor, weight: Tensor) -> Tensor:
        ctx.saved_sparseTensors = [input]
        # start = timeit.default_timer()
        out = spCpp.sparseLinear_forward(input.cTensor, weight)
        # print('linear_forward ', timeit.default_timer() - start)
        return out
    
    @staticmethod
    def backward(ctx, grad: Tensor) -> [None, SparseTensor]:
        input, = ctx.saved_sparseTensors
        # start = timeit.default_timer()
        flatten_input = input.reshape([input.shape[0], -1]).coalesce()
        grad_weight = spCpp.linear_backward_coo(flatten_input.cTensor, grad)
        # print('linear_backward ', timeit.default_timer() - start)
        return None, grad_weight

    
class SparseLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        # bias: bool = True,
        device=None, 
        dtype=None
    ):
        super().__init__(in_features, out_features, None, device, dtype)

    def forward(self, input: SparseTensor):
        return SparseLinearFunction.apply(input, self.weight)