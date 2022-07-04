from typing import Optional, List, Tuple, Union
from torch import nn, Tensor
from torch.autograd import Function
import torch

import sparseOps_cpp as spCpp
from .sparseTypes import SparseTensor, SparseParameter


class SparseLinearFunction(Function):
    @staticmethod
    def forward(ctx, input: SparseTensor, weight: Tensor) -> Tensor:
        ctx.saved_sparseTensors = [input]
        return spCpp.sparseDenseMM_forward(input.cTensor, weight)
    
    @staticmethod
    def backward(ctx, grad: Tensor) -> [None, SparseTensor]:
        input, = ctx.saved_sparseTensors
        grad_weight = spCpp.sparseDenseMM_backward(input.cTensor, grad)
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