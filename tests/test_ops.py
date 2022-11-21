from init import *


class TestOps(unittest.TestCase):
    
    def test_cat(self):
        # test when cat along sparse_dim
        input = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=1)
        other = SparseTensor.from_dense(torch.randn(4, 4), 2, sparse_dim=1)
        output1 = cat([input, other], dim=1).to_dense()
        output2 = torch.cat([input.to_dense(), other.to_dense()], dim=1)
        assert_close(output1, output2)
        # test when cat not along sparse_dim
        output1 = cat([input, other], dim=0).to_dense()
        output2 = torch.cat([input.to_dense(), other.to_dense()], dim=0)
        assert_close(output1, output2)

    def test_reshape(self):
        torch.manual_seed(0)
        input = SparseTensor.randn((64, 4, 8), 2, 1)
        output1 = input.reshape([2, -1, 2])
        output2 = input.to_dense().view(2, -1, 2)
        assert_close(output1.to_dense(), output2)

    # def test_max_pool(self):
    #     input = SparseTensor.from_dense(torch.randn(4, 4, 8, 8), 2, sparse_dim=1)
    #     output1 = max_pool2d(input, [2, 2]).to_dense()
    #     output2 = F.max_pool2d(input.to_dense(), [2, 2])
    #     assert_close(output1, output2)

    def test_topk(self):
        topk = 16
        input = SparseTensor.randn((64, 128, 8, 8), 32, sparse_dim=1, variant_len=True)
        output1 = input.topk(topk)
        self.assertEqual(output1.indices().shape[1], topk)
        vals, ids = input.to_dense().topk(topk, dim=1)
        output2 = torch.zeros_like(input.to_dense()).scatter(1, ids, vals)
        assert_close(output1.to_dense(), output2)

    def test_clamp_topk(self):
        input = SparseTensor.from_dense([[1, 2., 3], [0, -1, 3]], 3, 1)
        output1 = clamp_topk(input, 2, 1).to_dense()
        output2 = torch.tensor([[0, 2., 3], [0, 0, 3]])
        assert_close(output1, output2)
        input = SparseTensor.randn((64, 128, 8, 8), 32, sparse_dim=1)
        output1 = clamp_topk(input, k=16, dim=1).to_dense()
        vals, ids = input.to_dense().topk(16, dim=1)
        output2 = torch.zeros_like(input.to_dense()).scatter(1, ids, vals)
        assert_close(output1, output2)


class TestReduce(unittest.TestCase):

    def test_reduce_ops(self):
        input = SparseTensor.randn([32, 64, 8], 16, 1, variant_len=True) # try variant_len = True / False
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
        # check keepdim=True + partly reduce
        output1 = input.sum(0, True).to_dense()
        output2 = input.to_dense().sum(0, True)
        assert_close(output1, output2)
        # check partly reduce + count_nonzero
        output1 = input.count_nonzero(0).to_dense().to(torch.int64)
        output2 = input.to_dense().count_nonzero(0)
        assert_close(output1, output2)
        # check when multiple reduce axis are not contiguous
        input = SparseTensor.from_dense(torch.randn(10, 5, 4, 6, 3), 2, 2).coalesce()
        output1 = input.sum([0, 4], True).to_dense()
        output2 = input.to_dense().sum([0, 4], True)
        assert_close(output1, output2)
        # check when reduce large, should avoid stack overflow
        input = SparseTensor.from_dense(torch.randn(64000, 32), 16, 1).coalesce()
        output1 = input.sum([0], True).to_dense()
        output2 = input.to_dense().sum(0, True)
        assert_close(output1, output2)
        # # check full sparse_size
        # input = SparseTensor.from_dense(torch.randn(64, 512, 8, 8), 512, 1).coalesce()
        # output1 = input.count_nonzero([0, 2, 3])
        # output2 = input.to_dense().count_nonzero([0, 2, 3]).to(output1.dtype)
        # assert_close(output1.to_dense(), output2)
        
        # input.prod()


class TestElementwise(unittest.TestCase):

    def test_elementwise_ops(self):
        input = SparseTensor.randn((10, 5, 6, 3), 2, 1, variant_len=True)
        other = SparseTensor.randn((5, 6, 1), 3, 0, variant_len=True)
        # check when other is number
        output1 = (input + 1).to_dense()
        output2 = (input.to_dense() + 1) * input.mask() # should ignore zero entries
        assert_close(output1, output2)
        # check when other is sparse tensor
        output1 = (input + other)
        output2 = (input.to_dense() + other.to_dense()) * (input.to_dense() != 0).to(input.dtype)
        assert_close(output1.to_dense(), output2)
        # check mul
        output1 = (input * other).to_dense()
        output2 = input.to_dense() * other.to_dense()
        assert_close(output1, output2)

    def test_elementwise_keep_coalesced(self):
        # check if after op, still keep coalesed.
        input = SparseTensor.from_dense(torch.randn(10, 5, 6, 3), 2, sparse_dim=1)
        other = SparseTensor.from_dense(torch.randn(5, 6, 1), 3, sparse_dim=0)
        output1 = (input.coalesce() + other.coalesce())
        assert(output1.is_coalesced())
        assert_close(output1.coalesce().indices(), output1.indices())
        assert_close(output1.coalesce().values(), output1.values())

    def test_elementwise_broadcast(self):
        """only non-zero entries will be applied
        """
        # +/- float
        input = SparseTensor.from_dense([[1., 2.], [3., 0.]], 2, 1).coalesce()
        output1 = (input + 1).to_dense()
        output2 = torch.tensor([[2., 3.], [4., 0.]])
        assert_close(output1, output2)
        # +/- [..., 1 (dense dim), ...]
        input = SparseTensor.randn([64, 32, 8], 4, 1, variant_len=True)
        other = input.sum(0, True)
        output1 = (input + other)
        output2 = (input.to_dense() + other.to_dense()) * input.mask()
        assert_close(output1.to_dense(), output2)
        # # check if broadast only on non-zero items # todo...
        # print(input.indices()[0])
        # print(input.values()[0])
        # print(output1.indices()[0])
        # assert_close(input.indices(), output1.coalesce().indices())
        
    def test_rdiv(self):
        # check __rtruediv__
        input = SparseTensor.from_dense([[1., 0.], [2, 3]], 2, 1)
        output1 = (1. / input).to_dense()
        output2 = torch.tensor([[1., 0.], [0.5, 1 / 3]])
        assert_close(output1, output2)

    def test_elementwise_with_coo(self):
        input = SparseTensor.randn((4, 4), 2, sparse_dim=1)
        other = SparseTensor.randn((4, 4), 2, sparse_dim=0).to_dense().to_sparse()
        # test add_ (restrict by input mask)
        input1 = input.clone()
        output1 = input1.add_(other).to_dense()
        assert_close(output1, input1.to_dense())
        output2 = input.to_dense() + other.to_dense() * input.mask()
        assert_close(output1, output2)

    def test_elementwise_with_coo_dense(self):
        # test add_ (+ dense_size as suffix)
        # input1: [sparse_dims (2) + dense_dims (2)]
        id_range = 64; sparse_size = 20; dense_sizes = [30, 42]
        ids1 = torch.randint(0, id_range, (42, sparse_size))
        vals1 = torch.randn(42, sparse_size, *dense_sizes)
        input = SparseTensor.create(ids1, vals1, 1, id_range)
        input1 = input.clone()
        # input2: [sparse_dims (3) + dense_dims (1)]
        nse = 32
        ids2 = torch.cat([torch.randint(0, i, (1, nse)) for i in [42, id_range, dense_sizes[0]]], dim=0)
        vals2 = torch.randn(nse, *dense_sizes[-1:])
        other = torch.sparse_coo_tensor(ids2, vals2, size=(42, id_range, *dense_sizes))
        
        output1 = input1.add_(other).to_dense()
        assert_close(output1, input1.to_dense())
        # output2 = input.to_dense() + other.to_dense() * input.mask()
        # assert_close(output1, output2)



if __name__ == '__main__':
    unittest.main()