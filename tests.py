from typing import Optional, List, Tuple, Union
import unittest
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import time
import torch
from torch.testing import assert_close

import sparseOps_cpp as spCpp
from src.python.sparseTypes import SparseTensor
from src.python.sparseConv import SparseConv2d, SparseConv2dFunction
from src.python.sparseBatchNorm import SparseBatchNorm2d
from src.python.sparseOps import sparse_cat
from src.python.sparseLinear import SparseLinear


torch.set_default_dtype(torch.double)

def approx_grad(forward, input: Union[SparseTensor, Tensor], eps=1e-6) -> SparseTensor:
    """approximate gradient for gradient check"""
    assert input.dtype == torch.double, "input must be double"
    assert isinstance(input, (SparseTensor, Tensor))
    is_sparse = isinstance(input, SparseTensor)
    in_values = input.values().clone() if is_sparse else input.clone()
    val_shape = in_values.shape
    flat_values = in_values.view(-1)
    out_grad = torch.zeros_like(flat_values)
    num_itr = flat_values.shape[0]
    for i in range(num_itr):
        start = time.time()

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
        out_grad[i] = (out1 - out2).sum() / (2*eps)

        if i % 100 == 0:
            print('{} / {}, elapse_per: {}s'.format(i, num_itr, time.time() - start))

    out_grad = out_grad.view(val_shape)
    return input.update_with(values=out_grad) if is_sparse else out_grad


class TestGradCheck(unittest.TestCase):
    def test_dense(self):
        input = torch.arange(0.1, 1.2, 0.1, dtype=torch.double)[:, None]
        input.requires_grad = True
        weight = torch.randn([1, 11], dtype=torch.double)
        grad1 = approx_grad(lambda x: torch.matmul(weight, x), input)
        grad2 = weight.transpose(0, 1)
        assert_close(grad1, grad2)

    def test_sparse_matmul(self):
        ids = torch.tensor(range(1, 11))
        vals = torch.arange(0.1, 1.1, 0.1, dtype=torch.double)
        input = SparseTensor.create(ids, vals, 0, 11)
        weight = torch.randn([1, 11], dtype=torch.double)
        grad1 = approx_grad(
            lambda x: torch.matmul(weight, x.to_dense()[:, None]), input).to_dense()
        grad2 = weight.squeeze() * input.mask()
        assert_close(grad1, grad2)

    # def test_dense_conv(self):
    #     in_channels = 16; out_channels = 32; stride = [2, 2]; padding = [1, 1]; kernel = [2, 2]
    #     input = torch.randn(16, in_channels, 4, 4)
    #     conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
    #     output = conv(input)
    #     output.backward()
    #     grad1 = conv.weight.grad
    #     grad2 = approx_grad(lambda x: conv(x), input)
    #     assert_close(grad1, grad2)


class TestSparseTensor(unittest.TestCase):
    def test_coalesce(self):
        input = SparseTensor.from_dense(torch.randn(64, 64, 16, 16), 16, 1)
        output = input.coalesce()
        assert_close(input.to_dense(), output.to_dense())


class TestSparseConv2d(unittest.TestCase):

    def test_init(self):
        kernel = [4, 4]; stride = [2, 2]; padding = [1, 1]; in_channels = 64; in_connects = 4; out_channels = 128
        conv = SparseConv2d(in_channels, in_connects, out_channels, kernel, stride, padding)
        # check rand init
        indices = conv.weight.data.indices()
        values = conv.weight.data.values()
        assert indices.max() < in_channels and indices.min() >= 0
    
    def test_reset_parameters(self):
        pass

    def test_unfold(self):
        input = torch.randn(64, 64, 16, 16)
        stride = [2, 2]
        padding = [1, 1]
        kernel = [4, 4]
        dilation = [1, 1]
        output1 = spCpp.unfold(input, spCpp.size_hw(kernel), spCpp.size_hw(padding), spCpp.size_hw(stride), spCpp.size_hw(dilation))
        output2 = F.unfold(input, kernel, dilation, padding, stride)
        assert_close(output1, output2)

    def test_forward_backward(self):
        in_channels = 64; out_channels = 128; stride = [2, 2]; padding = [1, 1]; kernel = [4, 4]
        input = SparseTensor.from_dense(torch.randn(64, in_channels, 16, 16), 16, 1)
        conv = SparseConv2d(in_channels, 4, out_channels, kernel, stride, padding)
        dense_weight = conv.weight.data.to_dense().clone().detach()
        dense_weight.requires_grad = True
        # check forward
        output1 = conv.forward(input)#.to_dense()
        output2 = F.conv2d(input.to_dense(), dense_weight, None, stride, padding)
        assert_close(output1.to_dense(), output2)
        # check backward
        gradient = SparseTensor.from_dense(torch.randn(*output1.shape), 32, 1)
        output1.backward(gradient)
        grad1 = conv.weight.grad.to_dense()
        conv.weight.grad = None
        output2.backward(gradient.to_dense())
        grad2 = dense_weight.grad
        assert_close(grad1, grad2)

    def test_grad_check(self):
        in_channels = 16; out_channels = 32; stride = [2, 2]; padding = [1, 1]; kernel = [2, 2]
        input = SparseTensor.from_dense(torch.randn(16, in_channels, 4, 4), 4, 1)
        conv = SparseConv2d(in_channels, 4, out_channels, kernel, stride, padding)
        weight = conv.weight.data
        convF = SparseConv2dFunction()
        # grad1 Function
        output0 = SparseConv2dFunction.forward(convF, input, weight, kernel, stride, padding)
        # exclude non-sense entries when value is 0
        out_grad = output0.update_with(values=(output0.values() != 0).to(output0.dtype))
        grad1 = SparseConv2dFunction.backward(convF, out_grad)[1].to_dense() * weight.mask()
        # grad2
        grad2 = approx_grad(
            lambda x: SparseConv2dFunction.forward(convF, input, x, kernel, stride, padding), weight).to_dense()
        assert_close(grad1, grad2)


class TestLinear(unittest.TestCase):

    def test_SparseLinear(self):
        input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
        linear = SparseLinear(128*8*8, 10)
        # check forward
        output1 = linear.forward(input)
        output2 = F.linear(input.to_dense().view(input.to_dense().shape[0], -1), linear.weight)
        assert_close(output1, output2)
        # check backward
        gradient = torch.randn(64, 10)
        output1.backward(gradient)
        grad1 = linear.weight.grad.to_dense()
        linear.weight.grad = None
        output2.backward(gradient)
        grad2 = linear.weight.grad
        assert_close(grad1, grad2)


class TestOps(unittest.TestCase):

    def test_sparse_dense_conversion(self):
        torch.manual_seed(2)
        input = torch.randn(64, 64, 16, 16)
        sparse1 = SparseTensor.from_dense(input, 16, 1)
        dense1 = sparse1.to_dense()
        sparse2 = SparseTensor.from_dense(dense1, 16, 1)
        dense2 = sparse2.to_dense()
        assert_close(dense1, dense2)

    def test_reduce_ops(self):
        # torch.manual_seed(2)
        input = SparseTensor.from_dense(torch.randn(5, 6, 3), 2, 1)
        # check when sparse_dim included
        output1 = input.sum([0, 1], True)#.to_dense()
        output2 = input.to_dense().sum([0, 1], True)
        assert_close(output1, output2)
        # check when sparse_dim not included
        output1 = input.sum([0, 2], True).to_dense()
        output2 = input.to_dense().sum([0, 2], True)
        assert_close(output1, output2)
        # check keepdim=False
        output1 = input.sum([0, 2]).to_dense()
        output2 = input.to_dense().sum([0, 2])
        assert_close(output1, output2)
        # check keepdim=True  + partly reduce
        output1 = input.sum(0, True).to_dense()
        output2 = input.to_dense().sum(0, True)
        assert_close(output1, output2)
        # check partly reduce + count_nonzero
        output1 = input.count_nonzero(0).to_dense().to(torch.int64)
        output2 = input.to_dense().count_nonzero(0)
        assert_close(output1, output2)
        # check large number reduce sum
        torch.manual_seed(0)
        input = SparseTensor.from_dense(torch.randn(12000000, 1), 1, 1)
        # output1 = input.sum(0, True).to_dense()
        output1 = input.sum(0, True).to_dense()
        output2 = input.to_dense().sum(dim=0, keepdim=True)
        assert_close(output1, output2)
        
        # input.prod()

    def test_elementwise_ops(self):
        input = SparseTensor.from_dense(torch.randn(10, 5, 6, 3), 2, sparse_dim=1)
        other = SparseTensor.from_dense(torch.randn(5, 6, 1), 3, sparse_dim=0)
        # check when other is number
        output1 = (input + 1).to_dense()
        output2 = (input.to_dense() + 1) * input.mask() # should ignore zero entries
        assert_close(output1, output2)
        # check when other is sparse tensor
        output1 = (input + other).to_dense()
        output2 = (input.to_dense() + other.to_dense()) * (input.to_dense() != 0).to(input.dtype)
        assert_close(output1, output2)
        # check mul
        output1 = (input * other).to_dense()
        output2 = input.to_dense() * other.to_dense()
        assert_close(output1, output2)
        # check divisor should include all entries of dividend
        other = input.sum((0,2,3), keepdim=True)
        output1 = (input / other).to_dense()
        output2 = input.to_dense() / other.to_dense()
        assert_close(output1, output2)

    def test_elementwise_broadcast(self):
        """only non-zero entries will be applied
        """
        input = SparseTensor.from_dense([[1., 2.], [3., 0.]], 2, 1)
        # +/- float
        output1 = (input + 1).to_dense()
        output2 = torch.tensor([[2., 3.], [4., 0.]])
        assert_close(output1, output2)
        # +/- [..., 1 (dense dim), ...]
        other = input.sum(0, True)
        output1 = (input + other).to_dense()
        output2 = torch.tensor([[5., 4], [7, 0]])
        assert_close(output1, output2)
        

    def test_rdiv(self):
        # check __rtruediv__
        input = SparseTensor.from_dense([[1., 0.], [2, 3]], 2, 1)
        output1 = (1. / input).to_dense()
        output2 = torch.tensor([[1., 0.], [0.5, 1 / 3]])
        assert_close(output1, output2)

    def test_elementwise_inplace(self):
        input = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=1)
        other = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=0).to_dense().to_sparse()
        # test add_ (restrict by input mask)
        input1 = input.clone()
        output1 = input1.add_(other).to_dense()
        assert_close(output1, input1.to_dense())
        output2 = (input.to_dense() + other.to_dense()) * input.mask()
        assert_close(output1, output2)

    def test_sparse_cat(self):
        # test when cat along sparse_dim
        input = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=1)
        other = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=1)
        output1 = sparse_cat([input, other], dim=1).to_dense()
        output2 = torch.cat([input.to_dense(), other.to_dense()], dim=1)
        assert_close(output1, output2)
        # test when cat not along sparse_dim
        output1 = sparse_cat([input, other], dim=0).to_dense()
        output2 = torch.cat([input.to_dense(), other.to_dense()], dim=0)
        assert_close(output1, output2)
        

def batchNorm2d_dense(input: Tensor, dim: Tuple[int, ...], eps: float):
    """
    :input [batch, channel, height, width]
    """
    input_mask = (input != 0).to(input.dtype)
    count = input.count_nonzero(dim)[None, :, None, None]
    mean = input.sum(dim=dim, keepdim=True) / count
    x_centered = (input - mean) * input_mask
    var = (x_centered**2).sum(dim=dim, keepdim=True) / count
    var_mask = (var != 0).to(var.dtype)
    normed_input = x_centered / ((var + eps)*var_mask).sqrt()
    return normed_input, mean.squeeze(), var.squeeze()


class TestBatchNorm(unittest.TestCase):

    def test_stats(self):
        input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
        dim = (0,2,3)
        count1 = input.count_nonzero(dim, True).to_dense().squeeze()
        count2 = input.to_dense().count_nonzero(dim).to(count1.dtype)
        assert_close(count1, count2)
        var1 = input.var(dim, True).to_dense().squeeze()
        output2, mean2, var2 = batchNorm2d_dense(input.to_dense(), dim, 1e-5)
        diff1 = ((input - input.sum(dim, True))).sum(dim, True).to_dense()
        diff2 = (((input.to_dense() - input.to_dense().sum(dim=dim, keepdim=True)))
            * (input.to_dense() != 0).to(input.dtype)).sum(dim, keepdim=True)
        assert_close(diff1, diff2)
        assert_close(var1, var2)

    def test_forward(self):
        input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8, requires_grad=True), 32, 1)
        bn = SparseBatchNorm2d(128)
        running_mean2 = bn.running_mean.clone()
        running_var2 = bn.running_var.clone()
        bn.training = True
        # training + sparse
        output1 = bn.forward(input).to_dense()
        # training + dense
        output2, mean2, var2 = batchNorm2d_dense(input.to_dense(), (0,2,3), bn.eps)
        assert_close(output1, output2)
        assert_close(bn.running_mean, mean2*bn.momentum)
        assert_close(bn.running_var, 1*(1-bn.momentum) + var2*bn.momentum)

    def test_with_dense(self):
        out_channels = 8
        input = SparseTensor.from_dense(torch.randn(4, out_channels, 4, 4), out_channels, 1)
        bn = SparseBatchNorm2d(out_channels)
        dense_input = input.to_dense().clone().detach()
        input.requires_grad, dense_input.requires_grad = True, True
        # forward
        output1 = bn.forward(input)
        output2 = F.batch_norm(dense_input, None, None, None, None, training=True)
        assert_close(output1.to_dense(), output2)
        # backward
        gradients = output1.update_with(values=(output1.values() > 0).to(output1.dtype))
        output1.backward(gradients)
        grad1 = input.grad.to_dense()
        output2.backward(gradients.to_dense().clone().detach())
        grad2 = dense_input.grad
        print(grad1[0, 0])
        print(grad2[0, 0])
        assert_close(grad1, grad2)

    def test_backward(self):
        out_channels = 32
        input = SparseTensor.from_dense(torch.randn(4, out_channels, 4, 4), 8, 1)
        input.requires_grad = True
        bn = SparseBatchNorm2d(out_channels)
        output0 = bn.forward(input)
        out_grad = output0.update_with(values=(output0.values() > 0).to(output0.dtype))
        output0.backward(out_grad)
        grad1 = input.grad.to_dense() * input.mask()
        grad2 = approx_grad(lambda x: bn.forward(x), input).to_dense()
        assert_close(grad1, grad2)


if __name__ == '__main__':
    unittest.main()