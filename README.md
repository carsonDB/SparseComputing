# Sparse computing extension for Pytorch
This extension is about specified sparse tensors and related sparse computing operators (*CPU version*).

## Define: Single dimensional sparse tensor (SD format)
SparseTensor defined here has a single sparse dimension with rest dense dimensions, in short SD format.
For example, shape of image inputs is `[batch_size, channels (sparse), height, width]`.
The second dimension (channels) is sparse.
And this sparse tensor consists of two sub-tensors, ids and values (same shape like before), and an index indicates which dimension is sparse.
In Pytorch, [COO sparse tensor](https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors)
is more generalized case of this type .
But SD format would be more efficient on computing, theoretically.
Practically, it is defined as class `SparseTensor` in `./src/cpp/sparseTensor.h`

## Source code structure
- `src/cpp`: (C++) core implements.
- `src/python`: (Python) wrappers of C++ core and helpers for python world.

## How to install
At root directory, execute
```
python setup.py install
```

*Only tested in linux (wsl) environment.*

## How to use
In `src/python`, here provides `SparseConv2d`, `SparseBatchNorm` and `SparseLinear` layers.
Rest files in `src/python` provide other useful tools.
All test cases (below) are examples to follow.

## How to test
In `./tests` directory, all test cases can be executed with

```
python tests/{test_xxx.py}
```

For example, `python tests/test_ops.py`.

## Reference
- [Pytorch sparse official document](https://pytorch.org/docs/stable/sparse.html)
- [Extension: pytorch_sparse](https://github.com/rusty1s/pytorch_sparse)