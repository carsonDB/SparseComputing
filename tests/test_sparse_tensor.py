from init import *


class TestSparseTensor(unittest.TestCase):
    def setUp(self):
        self.input = SparseTensor.randn((64, 64, 16, 16), 16, 1)

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

    def test_func_override(self):
        input = self.input
        out1 = torch.zeros_like(input, memory_format=torch.preserve_format)
        assert_close(out1.shape, input.shape)


if __name__ == '__main__':
    unittest.main()
    