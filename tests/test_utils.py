from init import *


class TestUtils(unittest.TestCase):
    def test_sorted_merge(self):
        ids = torch.tensor([[0, 1], [5, 2]], dtype=torch.int64)
        out_ids, _ = spCpp.sorted_merge(ids, {0, 1}, 0)
        assert_close(out_ids, torch.tensor([[0], [1], [2], [5]]))
        # more practical case
        input = SparseTensor.randn((32, 64, 128, 8, 8), 32, 2, variant_len=True)
        output1 = spCpp.sorted_merge(input.indices(), {1, 2, 3, 4}, 2)[0]
        # assert_close(output1.to_dense(), output1.to_dense()) # todo...
        # check when multiple reduce axis are not contiguous
        input = SparseTensor.randn((10, 5, 4, 6, 3), 2, 2, variant_len=True)
        output1 = spCpp.sorted_merge(input.indices(), {0, 2, 4}, 2)[0].squeeze()
        output2 = input.indices().transpose(0, 1).transpose(3, 4).reshape(5, -1, 6).sort(1)[0]
        assert_close(output1, output2)


if __name__ == '__main__':
    unittest.main()