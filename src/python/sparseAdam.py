from typing import Optional, List, Tuple, Union
import torch


class SparseAdam(torch.optim.SparseAdam):
    """Inherit from: 
        SparseAdam: https://pytorch.org/docs/stable/_modules/torch/optim/sparse_adam.html#SparseAdam
        F.sparse_adam: https://github.com/pytorch/pytorch/blob/master/torch/optim/_functional.py
    """
    def __init__(self, params: Union[Tensor, SparseTensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        pass # todo...