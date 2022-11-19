# Sparse computing extension for Pytorch
Sparse tensors and related sparse computing ops (*CPU version*).

## Code structure
- `src/cpp`: (C++) core implements.
- `src/python`: (Python) wrappers of C++ core and helpers for python world.

## How to install
At root directory, execute
```python setup.py install```

*Only tested in linux (wsl) environment.*

## How to test
In `./tests` directory, all test cases can be executed with

`python tests/{test_xxx.py}`

For example, `python tests/test_ops.py`.

## Reference
- [Pytorch sparse document](https://pytorch.org/docs/stable/sparse.html)
- [Extension: pytorch_sparse](https://github.com/rusty1s/pytorch_sparse)