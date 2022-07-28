from typing import List
import torch
from torch.nn import functional as F

import sparseOps_cpp as spCpp
from .utils import norm_tuple, size_2_t
from .sparseTypes import SparseTensor


def cat(inputs: List[SparseTensor], dim: int = 0):
    sparse_dims = [i.sparse_dim() for i in inputs]
    assert len(sparse_dims) > 0 and sparse_dims.count(sparse_dims[0]) == len(sparse_dims),  \
        "all input sparse dim should be same."
    sparse_dim = sparse_dims[0]

    val_lst = [i.values() for i in inputs]
    out_vals = torch.cat(val_lst, dim=dim)
    if sparse_dim != dim:
        id_lst = [i.indices() for i in inputs]
        return inputs[0].update_with(indices=torch.cat(id_lst, dim=dim), values=out_vals)
    else:
        all_range = 0
        id_lst = []
        for t in inputs:
            ids = t.indices() + all_range
            all_range += t.range()
            id_lst.append(ids)

        out_ids = torch.cat(id_lst, dim=dim)
        return SparseTensor.create(out_ids, out_vals, dim, all_range)


def max_pool2d_v0(input: SparseTensor, kernel_size: size_2_t, **kwargs):
    kernel_size = norm_tuple(kernel_size, 2)
    vals, pool_ids = F.max_pool2d(input.values(), kernel_size, **kwargs, return_indices=True)
    ids = input.indices()
    pool_id_shape = pool_ids.shape
    id_shape = ids.shape
    ids = ids.view(id_shape[0], id_shape[1], -1)
    ids = ids.gather(-1, pool_ids.view(pool_id_shape[0], pool_id_shape[1], -1)).view_as(vals)
    return input.update_with(indices=ids, values=vals)


def conv_length(length: int, kernel: int, padding: int, stride: int, dilation: int = 1) -> int:
    return (length + 2*padding - (dilation * (kernel - 1) + 1)) // stride + 1

def max_pool2d(input: SparseTensor, kernel_size: size_2_t):
    assert input.sparse_dim() == 1
    assert input.indices().dim() == 4 and input.values().dim() == 4
    kernel_size = norm_tuple(kernel_size, 2)
    stride = norm_tuple(2, 2)
    padding = norm_tuple(0, 2)
    dilation = norm_tuple(1, 2)
    batch_size, sub_channels, in_height, in_width = input.indices().shape
    out_height = conv_length(in_height, kernel_size[0], padding[0], stride[0], dilation[0])
    out_width = conv_length(in_width, kernel_size[1], padding[1], stride[1], dilation[1])
    unfolded_shape = [batch_size, sub_channels, kernel_size[0] * kernel_size[1], out_height, out_width]
    # todo... unfold: Floating point exception
    unfolded_ids = spCpp.unfold(input.indices(), kernel_size, padding, stride, dilation).view(unfolded_shape)
    unfolded_vals = spCpp.unfold(input.values(), kernel_size, padding, stride, dilation).view(unfolded_shape)
    # unfolded_shape: [batch, sub_channels, kernel_height * kernel_width, out_height, out_width]
    unfolded_input = SparseTensor.create(unfolded_ids, unfolded_vals, 1, input.range()).coalesce()
    # [batch, sub_channels, out_height, out_width]
    return unfolded_input.max(2)


def clamp_topk(input: SparseTensor, k: int, dim: int):
    """keep topk along one dimension, other zeros.
        Shape is not changed, unlike `topk`.
    """
    assert dim == input.sparse_dim(), 'only support when dim = sparse_dim'
    vals, ids = input.values().topk(k, dim)
    ids = input.indices().gather(dim, ids)
    return input.update_with(indices=ids, values=vals)