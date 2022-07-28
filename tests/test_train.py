from init import *


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        sparseTensor = SparseTensor.from_dense([[1, 2.], [3, -1]], 1, 1)
        denseTensor = torch.tensor([[1, 2.], [3, -1]])
        self.params = [SparseParameter(sparseTensor), nn.Parameter(denseTensor)]
        self.lr = 1e-4
        self.optimizer = SparseAdam(self.params, lr=self.lr)

    def test_zero_grad(self):
        for p in self.params:
            p.grad = torch.sparse_coo_tensor([[0, 1], [1, 1]], [4, -9.], p.shape)
        self.optimizer.zero_grad(set_to_none=True)
        print(self.params[0].grad)

    def test_step(self):
        for p in self.params:
            p.grad = torch.sparse_coo_tensor([[0, 1], [1, 1]], [4, -9.], p.shape)
        self.optimizer.step()
        beta1 = 0.9; beta2 = 0.999; eps = 1e-8
        def update(param, grad):
            m = 0.; v = 0.
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_corr = m / (1 - beta1)
            v_corr = v / (1 - beta2)
            grad = m_corr / (math.sqrt(v_corr) + eps) 
            return param - self.lr * grad

        assert_close(self.params[0].data.to_dense(), torch.tensor([[0, update(2, 4)], [3, 0]]))
        assert_close(self.params[1].data, torch.tensor([[1, update(2, 4)], [3, update(-1, -9)]]))
        # check beta1, beta2

        # todo...



class TestAutograd(unittest.TestCase):
    def test_conv_bn(self):
        in_channels = 8; out_channels = 16; stride = [2, 2]; padding = [1, 1]; kernel = [4, 4]
        input = SparseTensor.from_dense(torch.randn(2, in_channels, 4, 4), 4, 1)
        conv = SparseConv2d(in_channels, 4, out_channels, kernel, stride, padding)
        bn = SparseBatchNorm2d(out_channels)
        output1 = bn(conv(input))
        gradient = (output1 > 0).to(output1.dtype)
        output1.backward(gradient)
        grad1 = conv.weight.grad.to_dense().clone() #* conv.weight.data.mask()
        def fn(w):
            conv.weight.data = w
            return bn(conv(input))
        grad2 = approx_grad(fn, conv.weight.data, mask=gradient.to_dense()).to_dense()
        print(grad1[0,0])
        print(grad2[0,0])
        assert_close(grad1, grad2)
        # todo...



if __name__ == '__main__':
    unittest.main()