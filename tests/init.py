from typing import Optional, List, Tuple, Union
import os
import sys
from torch.nn import functional as F
import torch
import timeit
import numpy as np
import unittest
from torch.testing import assert_close
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.python import *
import sparseOps_cpp as spCpp

torch.set_default_dtype(torch.double)


def approx_grad(forward, input: Union[SparseTensor, Tensor], mask: Union[Tensor, None] = None, eps=1e-6) -> SparseTensor:
    """approximate gradient for gradient check
        mask: same shape with output of forward.
    """
    assert input.dtype == torch.double, "input must be double"
    assert isinstance(input, (SparseTensor, Tensor))
    is_sparse = isinstance(input, SparseTensor)
    in_values = input.values().clone() if is_sparse else input.clone()
    val_shape = in_values.shape
    flat_values = in_values.view(-1)
    out_grad = torch.zeros_like(flat_values)
    num_itr = flat_values.shape[0]
    for i in range(num_itr):
        start = timeit.default_timer()

        in_plus = flat_values.clone()
        in_plus[i] += eps
        out1 = forward(input.update_with(values=in_plus.view(val_shape)) if is_sparse else in_plus)
        if isinstance(out1, SparseTensor):
            out1 = out1.to_dense()
        in_minus = flat_values.clone()
        in_minus[i] -= eps
        out2 = forward(input.update_with(values=in_minus.view(val_shape)) if is_sparse else in_minus)
        if isinstance(out2, SparseTensor):
            out2 = out2.to_dense()
        mask = mask if mask is not None else 1.
        out_grad[i] = ((out1 - out2) * mask).sum() / (2*eps)

        if i % 100 == 0:
            print('{} / {}, elapse_per: {}s'.format(i, num_itr, timeit.default_timer() - start))

    out_grad = out_grad.view(val_shape)
    return input.update_with(values=out_grad) if is_sparse else out_grad
