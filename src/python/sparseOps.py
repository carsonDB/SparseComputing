from typing import List
import torch
from torch.nn import functional as F
from .sparseTypes import SparseTensor


def sparse_cat(inputs: List[SparseTensor], dim: int):
    sparse_dims = [i.sparse_dim() for i in inputs]
    assert len(sparse_dims) > 0 and sparse_dims.count(sparse_dims[0]) == len(sparse_dims),  \
        "all input sparse dim should be same."
    sparse_dim = sparse_dims[0]

    val_lst = [i.values() for i in inputs]
    out_vals = torch.cat(val_lst, dim=dim)
    if sparse_dim != dim:
        id_lst = [i.indices() for i in inputs]
        return inputs[0].update_with(indices=torch.cat(id_lst, dim=dim), values=out_vals)
    else:
        all_range = 0
        id_lst = []
        for t in inputs:
            ids = t.indices() + all_range
            all_range += t.range()
            id_lst.append(ids)

        out_ids = torch.cat(id_lst, dim=dim)
        return SparseTensor.create(out_ids, out_vals, dim, all_range)
    

def max_pool2d(input: SparseTensor, **kwargs):
    return input.update_with(values=F.max_pool2d(input.values(), **kwargs))