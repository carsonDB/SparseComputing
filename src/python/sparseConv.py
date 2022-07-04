from typing import Optional, List, Tuple, Union
import math
from torch import nn, Tensor
from torch.autograd import Function
import time
import torch

import sparseOps_cpp as spCpp
from .sparseTypes import SparseParameter, SparseTensor
from .utils import size_2_t, norm_tuple
from .autograd import SparseFunction


class SparseConv2dFunction(SparseFunction):
    @staticmethod
    def forward(ctx, input: SparseTensor, weight: SparseTensor, kernel, stride, padding):
        outputs, unfolded_input = spCpp.sparseConv2d_forward(input.cTensor, weight.cTensor, stride, padding)
        ctx.saved_sparseTensors = [unfolded_input, kernel]
        return SparseTensor(outputs)

    @staticmethod
    def backward(ctx, grad: SparseTensor):
        unfolded_input, kernel = ctx.saved_sparseTensors
        grad_weight = spCpp.sparseConv2d_backward(grad.cTensor, unfolded_input, kernel)
        return None, grad_weight, None, None, None


class SparseConv2d(nn.Module):
    """each neuron (in_channel), has equal sparsity
    random sample along [in_channel], not include [height, width]
    
    input shape structure:
        indices: [batch, channels(sparse), height, width]
        values: [batch, channels(sparse), height, width]

    weight shape structure:
        indices: [out_channels, in_connects(sparse)]
        values: [out_channels, in_connects(sparse)] + [kernel_height, kernel_width]
    
    """

    def __init__(
        self, 
        in_channels: int,
        in_connects: int, # not include [kernel_h, kernel_w]
        out_channels: int,
        kernel_size: size_2_t, 
        stride: size_2_t = 1, 
        padding: size_2_t = 0
        # no bias currently
    ) -> None:        
        super().__init__()
        self.in_connects = in_connects
        # normalize stride / padding
        self.kernel_size = norm_tuple(kernel_size, 2)
        self.stride = norm_tuple(stride, 2)
        self.padding = norm_tuple(padding, 2)
        # create empty sparse weight
        indices = torch.zeros([out_channels, in_connects], dtype=torch.int64)
        values = torch.zeros([out_channels, in_connects, *self.kernel_size], dtype=torch.float)
        weightTensor = SparseTensor.create(indices, values, sparse_dim=1, range=in_channels, requires_grad=True)
        self.weight = SparseParameter(weightTensor)
        # self.bias = nn.Parameter(Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        values = self.weight.data.values()
        stdv = 1 / math.sqrt(self.in_connects * math.prod(self.kernel_size)) # he_uniform
        with torch.no_grad():
            self.weight.data.init_rand_indices()
            values.uniform_(-stdv, +stdv)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return SparseConv2dFunction.apply(input, self.weight.data, self.kernel_size, self.stride, self.padding)
