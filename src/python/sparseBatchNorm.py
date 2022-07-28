from typing import Optional, List, Tuple, Union
from torch import nn, Tensor
from torch.autograd import Function
import timeit
import torch

import sparseOps_cpp as spCpp
from .sparseTypes import SparseParameter, SparseTensor
from .utils import size_2_t, norm_tuple
from .autograd import SparseFunction


class SparseBatchNorm2dFunction(SparseFunction):
    """ Refer to: https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567
        temporarily not consider learnable variables beta/gamma
    """
    @staticmethod
    def forward(
        ctx,
        input: SparseTensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        is_training: bool,
        exponential_average_factor: float,
        eps: float,
    ) -> SparseTensor:
        dim = [0, 2, 3]
        # start = timeit.default_timer()
        input = input.coalesce()
        if is_training:
            count = input.count_nonzero(dim, True)
            mean = input.sum(dim, True) / count # count is zeros???
            x_centered = input - mean
            var = (x_centered ** 2).sum(dim, True) / count
        else:
            mean = input.mask_on_dense(running_mean[None, :, None, None].expand(*input.shape))
            x_centered = input - mean
            var = x_centered.mask_on_dense(running_var[None, :, None, None].expand(*x_centered.shape))
        
        # std = (var + eps).sqrt()
        std = var.update_with(values=(var.values() + eps).sqrt()) # todo...
        x_norm = x_centered / std
        # assert torch.isnan(x_norm.values()).sum().item() == 0, 'is_training: ' + str(is_training)
        
        if is_training:
            ctx.save_for_backward(x_centered, x_norm, std, count)
            # update running_... if training
            mean = mean.squeeze([0, 2, 3])
            mean.mask_on_dense(running_mean).add_to_dense(running_mean, alpha=-exponential_average_factor)
            mean.add_to_dense(running_mean, alpha=exponential_average_factor)
            var = var.squeeze([0, 2, 3])
            var.mask_on_dense(running_var).add_to_dense(running_var, alpha=-exponential_average_factor)
            var.add_to_dense(running_var, alpha=exponential_average_factor)
        
        # print(">> sparseBatchNorm_forward", timeit.default_timer() - start)
        return x_norm

    @staticmethod
    def backward(ctx, grad: SparseTensor):
        dim = [0, 2, 3]
        # start = timeit.default_timer()
        x_centered, x_norm, std, count = ctx.saved_tensors
        d_input = 1. / count / std * (
            count * grad - 
            grad.sum(dim, keepdim=True) - 
            x_norm * (grad * x_norm).sum(dim, keepdim=True))
        
        # print('>> sparseBatchNorm_backward', timeit.default_timer() - start)
        return d_input, None, None, None, None, None


class SparseBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        # affine=True, # not implemented
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps=eps, momentum=momentum, affine=False, 
            track_running_stats=track_running_stats, **factory_kwargs)

    def forward(self, input: SparseTensor) -> SparseTensor:
        exponential_average_factor = self.momentum if self.momentum is not None else 0.0
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return SparseBatchNorm2dFunction.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

