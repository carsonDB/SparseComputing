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
from src.python.sparseLinear import SparseLinear


def approx_grad(forward: float, input: SparseTensor, eps=1e-6) -> SparseTensor:
    """approximate gradient for gradient check"""
    assert isinstance(input, SparseTensor)
    in_values = input.values().clone()#.detach()
    val_shape = in_values.shape
    flat_values = in_values.view(-1)
    out_grad = torch.zeros_like(flat_values)
    num_itr = flat_values.shape[0]
    for i in range(num_itr):
        start = time.time()

        in_plus = flat_values.clone()
        in_plus[i] += eps
        out1 = forward(input.update_with(values=in_plus.view(val_shape)))
        if isinstance(out1, SparseTensor):
            out1 = out1.to_dense()
        in_minus = flat_values.clone()
        in_minus[i] -= eps
        out2 = forward(input.update_with(values=in_minus.view(val_shape)))
        if isinstance(out2, SparseTensor):
            out2 = out2.to_dense()
        out_grad[i] = (out1 - out2).sum() / (2*eps)

        if i % 100 == 0:
            print('{} / {}, elapse_per: {}s'.format(i, num_itr, time.time() - start))

    out_grad = out_grad.view(val_shape)
    return input.update_with(values=out_grad)


class TestGradCheck(unittest.TestCase):
    def test(self):
        ids = torch.tensor(range(1, 11))
        vals = torch.arange(0.1, 1.1, 0.1, dtype=torch.float)
        input = SparseTensor.create(ids, vals, 0, 11)
        weight = torch.randn([1, 11], dtype=torch.float)
        grad1 = approx_grad(
            lambda x: torch.matmul(weight, x.to_dense()[:, None]), input, 1e-6).to_dense()
        grad2 = weight.squeeze()
        grad1 *= input.mask()
        grad2 *= input.mask()
        print(grad1 - grad2)
        assert_close(grad1, grad2)


class TestSparseConv2d(unittest.TestCase):

    def test_init(self):
        kernel = [4, 4]
        stride = [2, 2]
        padding = [1, 1]
        in_channels = 64
        in_connects = 4
        out_channels = 128
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
        dense_weight = torch.tensor(conv.weight.data.to_dense(), requires_grad=True)
        # check forward
        output1 = conv.forward(input)#.to_dense()
        output2 = F.conv2d(input.to_dense(), dense_weight, None, stride, padding)
        assert_close(output1.to_dense(), output2)
        # check backward
        gradient = SparseTensor.from_dense(torch.randn(64, out_channels, 8, 8), 32, 1)
        output1.backward(gradient)
        grad1 = conv.weight.grad.to_dense()
        output2.backward(gradient.to_dense())
        grad2 = dense_weight.grad
        assert_close(grad1, grad2)

    def test_grad_check(self):
        in_channels = 16; out_channels = 32; stride = [2, 2]; padding = [1, 1]; kernel = [2, 2]
        input = SparseTensor.from_dense(torch.randn(16, in_channels, 4, 4), 4, 1)
        conv = SparseConv2d(in_channels, 4, out_channels, kernel, stride, padding)
        convF = SparseConv2dFunction()
        weight = conv.weight.data
        # grad1
        output0 = SparseConv2dFunction.forward(convF, input, weight, kernel, stride, padding)
        out_grad = output0.clone()
        out_grad.values().fill_(1)
        grad1 = SparseConv2dFunction.backward(convF, out_grad)[1].to_dense() * weight.mask()
        # grad2
        grad2 = approx_grad(
            lambda x: SparseConv2dFunction.forward(convF, input, x, kernel, stride, padding), weight).to_dense()
        print(grad1[0])
        print(grad2[0])
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
        input = SparseTensor.from_dense(torch.randn(1200000, 1), 1, 1)
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
        output2 = input.to_dense() + other.to_dense()
        assert_close(output1, output2)
        # check mul
        output1 = (input * other).to_dense()
        output2 = input.to_dense() * other.to_dense()
        assert_close(output1, output2)
        # check divisor should include all entries of dividend
        other = input.sum((0,2,3), keepdim=True)
        output1 = (input / other).to_dense()
        output2 = input.to_dense() / other.to_dense()
        # assert_close(output1, output2)

        # todo... elementwise, coo_tensor


def batchNorm2d_dense(input: Tensor, dim: Tuple[int, ...], eps: float):
    """
    :input [batch, channel, height, width]
    """
    count = input.count_nonzero(dim)[None, :, None, None]
    mean = input.sum(dim=dim, keepdim=True) / count
    var = ((input - mean)**2).sum(dim=dim, keepdim=True) / count
    normed_input = (input - mean) / (var + eps).sqrt()
    return normed_input, mean.squeeze(), var.squeeze()


class TestBatchNorm(unittest.TestCase):

    def test_stats(self):
        input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
        dim = (0,2,3)
        count1 = input.count_nonzero(dim, True).to_dense().squeeze()
        count2 = input.to_dense().count_nonzero(dim).to(torch.float)
        assert_close(count1, count2)
        var1 = input.var(dim, True).to_dense().squeeze()
        output2, mean2, var2 = batchNorm2d_dense(input.to_dense(), dim, 1e-5)
        diff1 = ((input - input.sum(dim, True))).sum(dim, True).to_dense()
        diff2 = ((input.to_dense() - input.to_dense().sum(dim=dim, keepdim=True))).sum(dim, keepdim=True)
        assert_close(diff1, diff2)
        # assert_close(var1, var2)

    def test_forward(self):
        input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
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

    def test_backward(self):
        input = SparseTensor.from_dense(torch.randn(4, 32, 4, 4), 8, 1)
        input.requires_grad = True
        bn = SparseBatchNorm2d(128)
        output0 = bn.forward(input)
        out_grad = output0.clone()
        out_grad.values().fill_(1)
        output0.backward(out_grad)
        grad1 = input.grad.to_dense() * input.mask()
        grad2 = approx_grad(lambda x: bn.forward(x), input).to_dense()
        print(grad1[0, 0])
        print(grad2[0, 0])
        assert_close(grad1, grad2)


if __name__ == '__main__':
    unittest.main()