from init import *


class TestSparseTensor(unittest.TestCase):
    def setUp(self):
        self.input = SparseTensor.randn((64, 64, 16, 16), 16, 1)

    def test_to_dense(self):
        id_range = 2; sparse_size = 1; dense_sizes = [3]
        ids1 = torch.tensor([[0], [1]]) # [2, 1 (sparse)]
        vals1 = torch.tensor([[[0.1, 0.3, 0.4]], [[-0.3, 0, 3]]])
        input = SparseTensor.create(ids1, vals1, 1, id_range)
        output1 = input.to_dense()
        output2 = torch.tensor([[[0.1, 0.3, 0.4], [0, 0, 0]], [[0, 0, 0], [-0.3, 0, 3]]], dtype=output1.dtype)
        assert_close(output1, output2)

    def test_sparse_dense_conversion(self):
        input = torch.randn(64, 64, 16, 16)
        sparse1 = SparseTensor.from_dense(input, 16, 1)
        dense1 = sparse1.to_dense()
        sparse2 = SparseTensor.from_dense(dense1, 16, 1)
        dense2 = sparse2.to_dense()
        assert_close(dense1, dense2)

    def test_coalesce(self):
        input = self.input
        assert_close(input.coalesce().to_dense(), input.to_dense())
        input = SparseTensor.from_dense(torch.randn(10, 5, 6, 3), 2, 1)
        assert_close(input.coalesce().to_dense(), input.to_dense())
        self.assertTrue(input.coalesce().is_coalesced())
        # case: channels=1
        input = SparseTensor.from_dense([[1.], [3]], 1, 1)
        self.assertTrue(input.coalesce().is_coalesced())
        # case: variant-len sorted arrays (sparse_dim)

        # todo... add reduntance and unordered

    def test_mask_on_dense(self):
        # todo... cases: when ids of sparseTensor has invalid entries (values are 0)
        pass

    def test_mask(self):
        id_range = 2; sparse_size = 1; dense_sizes = [3]
        ids1 = torch.tensor([[0], [1]]) # [2, 1 (sparse)]
        vals1 = torch.ones(2, 1, *dense_sizes)
        input = SparseTensor.create(ids1, vals1, 1, id_range)
        mask1 = input.mask()
        mask2 = torch.tensor([[[1, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]], dtype=mask1.dtype)
        assert_close(mask1, mask2)

    def test_backward_hook(self):
        input = SparseTensor.randn((64, 128, 8, 8), 16, 1)
        conv = SparseConv2d(128, 4, 256, 4, 2, 1)
        output1 = conv.forward(input)#.to_dense()
        gradient = SparseTensor.randn(output1.shape, 64, 1)
        output1.backward(gradient)
        grad1 = conv.weight.grad
        h = conv.weight.data.register_hook(lambda grad: grad * 2)
        output1.backward(gradient)
        grad2 = conv.weight.grad
        assert_close((grad1 * 2).to_dense(), grad2.to_dense())



    def test_func_override(self):
        input = self.input
        out1 = torch.zeros_like(input, memory_format=torch.preserve_format)
        assert_close(out1.shape, input.shape)


if __name__ == '__main__':
    unittest.main()
    