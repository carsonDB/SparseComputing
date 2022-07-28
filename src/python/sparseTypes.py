"""wrapper of cpp Sparse Classes"""
from typing import Optional, List, Tuple, Union
import functools
import torch
from torch import nn, Tensor

import sparseOps_cpp as spCpp
from .utils import size_n_t_none, size_n_t, norm_tuple
from .autograd import sparse_tensor_backward


HANDLED_FUNCTIONS = {}

class SparseTensor(object):
    """ sparse_tensor + dense_tensor
    """

    # def __new__(cls, cTensor):
    #     return super().__new__(cls)
        # return torch.sparse_coo_tensor(size=cTensor.values().shape).__new__(cls)
    
    def __init__(self, cTensor):
        self.cTensor = cTensor
        self.grad = None
        self.grad_fn = None

    @classmethod
    def create(cls, indices, values, sparse_dim, range, requires_grad=None):
        cTensor = spCpp.SparseTensor(indices, values, sparse_dim, range)
        # only values allow gradient adjustment.
        if (requires_grad is not None):
            cTensor.values().requires_grad = requires_grad
        return SparseTensor(cTensor)

    def indices(self):
        return self.cTensor.indices()
    
    def values(self):
        return self.cTensor.values()

    def sparse_dim(self):
        return self.cTensor.sparse_dim()

    def range(self):
        return self.cTensor.range()
    
    def is_coalesced(self):
        return self.cTensor.is_coalesced()

    def coalesce(self):
        return SparseTensor(self.cTensor.coalesce())

    def init_rand_indices(self):
        self.cTensor.init_rand_indices()
    
    def update_with(self, indices=None, values=None, sparse_dim=None):
        indices = indices if isinstance(indices, Tensor) else self.indices()
        values = values if isinstance(values, Tensor) else self.values()
        sparse_dim = sparse_dim if isinstance(sparse_dim, int) else self.sparse_dim()
        return SparseTensor(self.cTensor.update_with(indices, values, sparse_dim))

    @property
    def requires_grad(self):
        return self.values().requires_grad
    
    @requires_grad.setter
    def requires_grad(self, require: bool):
        self.values().requires_grad = require

    def squeeze(self, dim: size_n_t):
        dims = norm_tuple(dim, 1)
        sparse_dim = self.sparse_dim()
        ids = self.indices()
        vals = self.values()
        assert not (sparse_dim in dims), 'squeeze should not include sparse_dim'
        # from end to start
        for d in sorted(dims, reverse=True):
            ids = ids.squeeze(d)
            vals = vals.squeeze(d)
            if sparse_dim > d:
                sparse_dim -= 1
        return self.update_with(indices=ids, values=vals, sparse_dim=sparse_dim)

    def expand_indices_as_values(self):
        vals = self.values()
        ids = self.indices()
        # expand ids to be same with vals
        for i in range(ids.dim(), vals.dim()):
            ids = ids.unsqueeze(i)
        return ids.expand_as(vals)

    @classmethod
    def from_dense(cls, dense_tensor, k, sparse_dim):
        """from dense tensor
        :param k select topk along dim
        :return only contain sparse_shape
        """
        if isinstance(dense_tensor, (list, tuple, int, float)):
            dense_tensor = torch.tensor(dense_tensor)
        assert (not dense_tensor.is_sparse), "Tensor must be a strided (dense) tensor"
        vals, ids = dense_tensor.topk(k, sparse_dim, True, False)
        return cls.create(ids, vals, sparse_dim, dense_tensor.shape[sparse_dim])

    def to_dense(self):
        vals = self.values()
        ids = self.expand_indices_as_values()
        out_shape = list(vals.shape)
        out_shape[self.sparse_dim()] = self.range()
        out = torch.zeros(*out_shape, dtype=vals.dtype)
        out.scatter_add_(self.sparse_dim(), ids, vals)
        return out
    
    def mask_on_dense(self, dense: Tensor):
        assert isinstance(dense, Tensor) and not dense.is_sparse
        ids = self.indices()
        vals = dense.gather(self.sparse_dim(), ids) * (self.values() != 0).to(self.dtype)
        return self.update_with(indices=ids, values=vals)

    def mask(self):
        vals = self.values()
        ids = self.expand_indices_as_values()
        out_shape = list(vals.shape)
        out_shape[self.sparse_dim()] = self.range()
        out = torch.zeros(*out_shape, dtype=vals.dtype)
        out.scatter_add_(self.sparse_dim(), ids, vals.abs())
        return (out > 0).to(self.dtype)

    # @classmethod
    # def from_fake_coo_tensor(cls, input: Tensor):
    #     assert input.layout == torch.sparse_coo
    #     info = input.restricted_info
    #     ids = input._indices().view(info['indices_size'])
    #     vals = input._values().view(info['values_size'])
    #     return cls.create(ids, vals, info['sparse_dim'], info['range'])

    # def to_fake_coo_tensor(self):
    #     ids = self.indices().view(-1)[None, :]
    #     vals = self.values().view(*ids.shape[1:], -1)
    #     out = torch.sparse_coo_tensor(ids, vals)
    #     out.coalesce = lambda: {} # forbidden resort
    #     out.restricted_info = {
    #         'indices_size': self.indices().shape, 
    #         'values_size': self.values().shape, 
    #         'range': self.range(), 
    #         'sparse_dim': self.sparse_dim()
    #     }
    #     return out

    def to_coo_tensor(self):
        raise NotImplementedError

    def size(self, dim: Union[None, int] = None):
        # replace size[sparse_dim] with range
        if dim is not None and dim == self.sparse_dim():
            return self.range()
        elif dim is None :
            size = list(self.values().size())
            size[self.sparse_dim()] = self.range()
            return torch.Size(size)
        else:
            return self.values().size(dim)
    
    @property
    def shape(self):
        return self.size()

    def numel(self):
        acc = 1
        for i in self.size():
            acc *= i
        return acc

    @property
    def dtype(self):
        return self.values().dtype

    def __repr__(self):
        return "{}(indices={}, value={}, size={}, requires_grad={})" \
            .format(self.__class__.__name__, self.indices(), self.values(), tuple(self.size()), self.requires_grad)

    def add_(self, other: Tensor):
        if isinstance(other, Tensor):
            if other.layout == torch.sparse_coo:
                pass # todo...
                raise NotImplemented('sparse_coo todo...')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def add_to_dense(self, dense: Tensor, alpha=1):
        ids = self.expand_indices_as_values()
        vals = self.values() * alpha
        dense.scatter_add_(self.sparse_dim(), ids, vals)

    def __add__(self, other):
        return elementwise_ops(self, other, 'add')

    def __radd__(self, other):
        return elementwise_ops(self, other, 'add')

    def add_(self, other):
        return elementwise_ops(self, other, 'add', inplace=True)
    
    def __sub__(self, other):
        return elementwise_ops(self, other, 'sub')

    def sub_(self, other):
        return elementwise_ops(self, other, 'sub', inplace=True)
    
    def __mul__(self, other):
        return elementwise_ops(self, other, 'mul')
    
    def __rmul__(self, other):
        return elementwise_ops(self, other, 'mul')
    
    def mul_(self, other):
        return elementwise_ops(self, other, 'mul', inplace=True)

    def __truediv__(self, other):
        return elementwise_ops(self, other, 'div')

    def __rtruediv__(self, prev):
        mask = (self.values() != 0).to(self.dtype) # inf -> nan when div by 0
        if isinstance(prev, (int, float)):
            vals = (prev / self.values()) * mask
            vals[vals != vals] = 0 # nan -> 0. when div by 0
            return self.update_with(values=vals)
        raise NotImplementedError

    def __gt__(self, other):
        return elementwise_ops(self, other, 'gt')
    
    def __ge__(self, other):
        return elementwise_ops(self, other, 'ge')
    
    def __lt__(self, other):
        return elementwise_ops(self, other, 'lt')
    
    def __le__(self, other):
        return elementwise_ops(self, other, 'le')
    
    def __eq__(self, other):
        return elementwise_ops(self, other, 'eq')
    
    def div_(self, other):
        return elementwise_ops(self, other, 'div', inplace=True)
    
    def max(self, dim: size_n_t_none = None, keepdim=False):
        return reduce_ops(self, dim, keepdim, 'max')
    
    def min(self, dim: size_n_t_none = None, keepdim=False):
        return reduce_ops(self, dim, keepdim, 'min')

    def sum(self, dim: size_n_t_none = None, keepdim=False):
        return reduce_ops(self, dim, keepdim, 'sum')

    def count_nonzero(self, dim: size_n_t, keepdim=False):
        return reduce_ops(self, dim, keepdim, 'count_nonzero')

    def mean(self, dim: size_n_t, keepdim=False):
        return self.sum(dim, keepdim) / self.count_nonzero(dim, keepdim)
    
    def var(self, dim: size_n_t, keepdim=False):
        return ((self - self.mean(dim, keepdim)) ** 2).sum(dim, keepdim) / self.count_nonzero(dim, keepdim)
    
    def __pow__(self, other: float):
        return elementwise_ops(self, other, 'pow')

    def sqrt(self):
        return elementwise_ops(self, 0.5, 'pow')

    def __neg__(self):
        return self.update_with(values=-self.values())

    def backward(self, grad = None):
        if not isinstance(grad, SparseTensor):
            raise NotImplementedError
        sparse_tensor_backward(self, grad)

    def register_hook(self, fn):
        # todo...
        pass

    def clone(self):
        return self.update_with(indices=self.indices().clone(), values=self.values().clone())
    
    def detach(self):
        return self.update_with(indices=self.indices().detach(), values=self.values().detach())

    def to(self, *args, **kwargs):
        return self.update_with(values=self.values().to(*args, **kwargs))

    def reshape(self, shape: List[int]):
        acc = 1
        id = -1
        for i, s in enumerate(shape):
            if s == -1 and id == -1:
                id = i
            else:
                assert s > 0, 'only allow at most one -1 dimension'
                acc *= s
        
        out_shape = shape.copy()
        if id >= 0:
            out_shape[id] = self.numel() // acc

        t = spCpp.reshape(self.cTensor, out_shape)
        return SparseTensor(t)

    def contiguous(self):
        return self.update_with(indices=self.indices().contiguous(), values=self.values().contiguous())

    def topk(self, k):
        """topk along sparse_dim
        """
        assert self.indices().dim() == self.values().dim(), """
            only support scalar value, but indices.dim={}, values.dim={}""".format(
                self.indices().dim, self.values().dim)
        dim = self.sparse_dim()
        k = min(k, self.indices().shape[dim])
        vals, ids = self.values().topk(k, dim, largest=True, sorted=False)
        ids = self.indices().gather(dim, ids)
        return self.update_with(indices=ids, values=vals)

    @classmethod
    def randn(cls, shape, topk, sparse_dim, variant_len=False):
        """
            variant_len: have different length, with zeros tailing, along each sorted array (sparse_dim)
        """
        out = cls.from_dense(torch.randn(*shape), topk, sparse_dim)
        if not variant_len:
            return out.coalesce()
        ids_shape = list(out.indices().shape).copy()
        mask_shape = ids_shape.copy()
        mask_shape[sparse_dim] = 1
        arange_shape = [1] * sparse_dim + [-1] + [1] * (len(mask_shape) - sparse_dim - 1)
        mask = torch.arange(topk).view(arange_shape) < torch.randint(0, topk, mask_shape)
        out = out.update_with(indices=out.indices() * mask, values=out.values() * mask).coalesce()
        return out
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, SparseTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(torch_function):
    """Register a torch function override for SparseTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


@implements(torch.zeros_like)
def zeros_like(input: SparseTensor, **kwargs) -> Tensor:
    vals = input.values()
    out = torch.zeros(input.shape, dtype=input.dtype, layout=vals.layout, device=vals.device)
    if 'memory_format' in kwargs:
        out.memory_format = kwargs['memory_format']
    return out


def reduce_ops(
    input: SparseTensor,
    dim: size_n_t_none,
    keepdim: bool, 
    op: str
) -> Union[SparseTensor, Tensor]:
    if dim is None:
        dim = range(input.values().dim())
    dim = norm_tuple(dim, 1)
    # dense tensor
    if input.sparse_dim() in dim:
        return getattr(input.values(), op)(dim=dim, keepdim=keepdim)
    # sparse tensor
    assert hasattr(spCpp, 'reduce_' + op), 'not Implemented yet'
    cTensor = getattr(spCpp, 'reduce_' + op)(input.cTensor, set(dim), keepdim)
    return SparseTensor(cTensor)


def elementwise_ops(
    input: SparseTensor, 
    other: Union[SparseTensor, float, int], 
    op: str,
    inplace: bool = False
) -> SparseTensor:
    if (isinstance(other, SparseTensor)):
        assert inplace == False, 'not implemented yet'
        assert hasattr(spCpp, 'elementwise_' + op), 'not Implemented yet'
        cTensor = getattr(spCpp, 'elementwise_' + op)(input.cTensor, other.cTensor)
        return SparseTensor(cTensor)
    elif isinstance(other, (float, int)):
        suffix = '_' if inplace else ''
        mask = (input.values() != 0.).to(input.dtype)
        values = getattr(torch, op + suffix)(input.values(), other) * mask
        return input if inplace else input.update_with(values=values)
    elif isinstance(other, Tensor) and other.is_sparse:
        assert other.layout == torch.sparse_coo, 'not Implemented yet' # todo... have implemented ???
        assert hasattr(spCpp, 'elementwise_' + op), 'not Implemented yet'
        cTensor = getattr(spCpp, 'elementwise_' + op)(input.cTensor, other, inplace)
        return input if inplace else SparseTensor(cTensor)
    elif isinstance(other, Tensor) and not other.is_sparse:
        raise NotImplementedError('temporarily use `mask_on_dense` to convert dense to sparse before.')
    else:
        raise NotImplementedError



class SparseParameter(nn.Parameter):
    """wrapper of nn.Parameter specified for Sparse classes"""
    def __new__(cls, data):
        self = super().__new__(cls) # todo... maybe remove
        self._data = data
        return self
    
    # override tensor.data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def grad(self):
        return self._data.grad
    
    @grad.setter
    def grad(self, grad):
        self._data.grad = grad

    def size(self, dim = None):
        return self._data.size(dim)

    @property
    def shape(self):
        return self.size()
    
    def add_(self, other):
        return self._data.add_(other)
    
    def __repr__(self):
        return 'Parameter containing:\n' + self._data.__repr__()


