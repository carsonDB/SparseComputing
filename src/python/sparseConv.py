from typing import Optional, List, Tuple, Union
import math
from torch import nn, Tensor
from torch.autograd import Function
import timeit
import torch

import sparseOps_cpp as spCpp
from .sparseTypes import SparseParameter, SparseTensor
from .utils import size_2_t, norm_tuple
from .autograd import SparseFunction



class SparseConv2dFunction(SparseFunction):
    @staticmethod
    def forward(ctx, input: SparseTensor, weight: SparseTensor, kernel, stride, padding, grad_clamp=0):
        # start = timeit.default_timer()
        outputs, unfolded_input = spCpp.sparseConv2d_forward(input.cTensor, weight.cTensor, stride, padding)
        ctx.saved_sparseTensors = [unfolded_input, weight.cTensor, kernel, grad_clamp]
        # print(">> sparseConv2d_forward", timeit.default_timer() - start)
        return SparseTensor(outputs)

    @staticmethod
    def backward(ctx, grad: SparseTensor):
        unfolded_input, weightCTensor, kernel, grad_clamp = ctx.saved_sparseTensors
        # start = timeit.default_timer()
        out_channels_ids, weight_grads = spCpp.conv2d_backward(
            grad.cTensor, unfolded_input, weightCTensor, kernel)
        # mask on each filter
        grad_vals = weight_grads.values()
        grad_min = grad_vals.view(grad_vals.shape[0], -1).quantile(q=grad_clamp, dim=1, keepdim=True)
        mask = (grad_vals > grad_min.view(grad_vals.shape[0], 1, 1, 1)).type(grad_vals.dtype)
        grad_vals.mul_(mask)
        # convert to coo_sparse tensor
        coo_grads = SparseConv2d.weight_grads_to_coo_tensor(out_channels_ids, weight_grads)
        # coo_grads = spCpp.sparseConv2d_backward_v0(grad.cTensor, unfolded_input, kernel)
        # print('>> sparseConv2d_backward', timeit.default_timer() - start)

        return None, coo_grads, None, None, None


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
        padding: size_2_t = 0,
        grad_clamp: float = 0, # mask on each filter (in_connects, kernel_h, kernel_w)
        # no bias currently
        device=None,
        dtype=None
    ) -> None:        
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_connects = in_connects
        self.grad_clamp = grad_clamp
        # normalize stride / padding
        self.kernel_size = norm_tuple(kernel_size, 2)
        self.stride = norm_tuple(stride, 2)
        self.padding = norm_tuple(padding, 2)
        # create empty sparse weight
        indices = torch.zeros([out_channels, in_connects], dtype=torch.int64)
        values = torch.zeros([out_channels, in_connects, *self.kernel_size], **factory_kwargs)
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
        return SparseConv2dFunction.apply(
            input, self.weight.data, self.kernel_size, self.stride, self.padding, self.grad_clamp)

    @staticmethod
    def weight_grads_to_coo_tensor(out_channels_ids: torch.Tensor, weight_grads: spCpp.SparseTensor):
        """ out_c is part of out_channels
            - out_channels_ids: [out_c]
            - weight_grads: indices [out_c, in_c], values: [out_c, in_c, kernel_h, kernel_w]
        return:
            sparse_coo: indices: [2, out_c*in_c], values: [out_c*in_c], size: [out_c, range, kernel_h, kernel_w]
        """
        assert weight_grads.indices().dim() == 2
        assert weight_grads.values().dim() == 4
        w_shape = list(weight_grads.values().shape)
        out_c, in_c = w_shape[:2]
        nse = out_c * in_c
        dims = weight_grads.indices().dim()
        ids = torch.zeros(dims, nse, dtype=weight_grads.indices().dtype)
        vals = weight_grads.values().view(nse, *w_shape[2:])
        # fill ids
        ids[0] = out_channels_ids[:, None].expand([out_c, in_c]).reshape(-1)
        ids[1] = weight_grads.indices().view(-1)

        return torch.sparse_coo_tensor(ids, vals, [w_shape[0], weight_grads.range(), *w_shape[2:]])


