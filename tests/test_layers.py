from init import *


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


class TestConv(unittest.TestCase):

    def setUp(self):
        # self.in_channels = 64; out_channels = 512; stride = [2, 2]; padding = [1, 1]; kernel = [4, 4]
        # input = SparseTensor.from_dense(torch.randn(64, in_channels, 8, 8), 16, 1)
        pass

    def test_init(self):
        kernel = [4, 4]; stride = [2, 2]; padding = [1, 1]; in_channels = 64; in_connects = 4; out_channels = 128
        conv = SparseConv2d(in_channels, in_connects, out_channels, kernel, stride, padding)
        # check rand init
        indices = conv.weight.data.indices()
        values = conv.weight.data.values()
        assert indices.max() < in_channels and indices.min() >= 0
    
    def test_reset_parameters(self):
        # todo...
        pass

    def test_unfold(self):
        input = torch.randn(64, 64, 16, 16)
        stride = [2, 2]
        padding = [1, 1]
        kernel = [4, 4]
        dilation = [1, 1]
        output1 = spCpp.unfold(input, kernel, padding, stride, dilation)
        output2 = F.unfold(input, kernel, dilation, padding, stride)
        assert_close(output1, output2)

    def test_forward_backward(self):
        in_channels = 64; out_channels = 512; stride = [2, 2]; padding = [1, 1]; kernel = [4, 4]
        input = SparseTensor.randn((64, in_channels, 16, 16), 16, 1)
        conv = SparseConv2d(in_channels, 4, out_channels, kernel, stride, padding)
        dense_weight = conv.weight.data.to_dense().clone().detach()
        dense_weight.requires_grad = True
        # check forward
        output1 = conv.forward(input)#.to_dense()
        output2 = F.conv2d(input.to_dense(), dense_weight, None, stride, padding)
        assert_close(output1.to_dense(), output2)
        # check backward
        gradient = SparseTensor.randn(output1.shape, 32, 1)
        output1.backward(gradient)
        grad1 = conv.weight.grad.to_dense()
        conv.weight.grad = None
        output2.backward(gradient.to_dense())
        grad2 = dense_weight.grad
        assert_close(grad1 * conv.weight.data.mask(), grad2 * conv.weight.data.mask())
        # assert_close(grad1, grad2)

    def test_grad_check(self):
        # in_channels = 64; out_channels = 512; stride = [2, 2]; padding = [1, 1]; kernel = [4, 4]
        # input = SparseTensor.randn((64, in_channels, 8, 8), 16, 1)
        in_channels = 16; out_channels = 32; stride = [2, 2]; padding = [1, 1]; kernel = [2, 2]
        input = SparseTensor.randn((16, in_channels, 4, 4), 4, 1)
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
        # print(grad2[0, 0])
        assert_close(grad1, grad2)


class TestLinear(unittest.TestCase):

    def test_SparseLinear(self):
        torch.set_default_dtype(torch.float)
        input = SparseTensor.randn((64, 128, 8, 8), 32, 1)
        linear = SparseLinear(128 * 8 * 8, 10)
        linear_shape = [input.shape[0], -1]
        # check forward
        output1 = linear.forward(input)
        output2 = F.linear(input.to_dense().view(linear_shape), linear.weight)
        assert_close(output1, output2)
        # check backward
        gradient = torch.randn(64, 10)
        output1.backward(gradient)
        grad1 = linear.weight.grad.to_dense()
        linear.weight.grad = None
        output2.backward(gradient)
        grad2 = linear.weight.grad
        assert_close(grad1, grad2)


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
    def setUp(self):
        self.input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
        self.bn = SparseBatchNorm2d(128)

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
        # input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 16, 1)
        input = SparseTensor.randn((64, 128, 8, 8), 16, 1, variant_len=False)
        input.requires_grad = True
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

    def test_eval(self):
        # eval: only forward.
        dense_input = torch.tensor([[1, 2.], [3, 0]])[..., None, None]
        input = SparseTensor.from_dense(dense_input, 1, 1)
        bn = SparseBatchNorm2d(2)
        bn.training = False
        output1 = bn.forward(input).to_dense()
        std = torch.tensor(1 + bn.eps).sqrt().item()
        assert_close(output1, torch.tensor([[0, 2. / std], [3 / std, 0]])[..., None, None])

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
        assert_close(grad1, grad2) # todo...

    def test_backward(self):
        out_channels = 32
        input = SparseTensor.from_dense(torch.randn(4, out_channels, 4, 4), 8, 1)
        input.requires_grad = True
        bn = SparseBatchNorm2d(out_channels)
        output0 = bn.forward(input)
        out_grad = (output0 > 0).to(output0.dtype)
        output0.backward(out_grad)
        grad1 = input.grad.to_dense() * input.mask()
        grad2 = approx_grad(lambda x: bn.forward(x), input, mask=out_grad.to_dense()).to_dense()
        print(grad1[0, 0])
        print(grad2[0, 0])
        assert_close(grad1, grad2) # todo...



if __name__ == '__main__':
    unittest.main()