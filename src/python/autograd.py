from typing import Optional, List, Tuple, Union
import queue
import torch
from torch import Tensor
from torch.autograd import Function
from . import sparseTypes as spt


class Node():
    def __init__(self, ctx, backward_fn, output_nr):
        self.ctx = ctx
        self.backward_fn = backward_fn
        self.output_nr = output_nr
        self.next_functions: [{node: Function, input_nr: int}, ...] = []
    def __call__(self, *args):
        return self.backward_fn(self.ctx, *args)

class GradAcc(Node):
    # leaf variables
    def __init__(self, tensor):
        self.tensor = tensor
        self.next_functions = []

    def __call__(self, grad):
        # sparse tensor cannot be accumulated
        self.tensor.grad = grad


class SparseFunction():
    """Support for self-defined SparseTensor.
    Currently testing only on SparseConv / SparseBatchNorm2d layer.
    """

    def save_for_backward(self, *args):
        self.to_save = tuple(args)

    @property
    def saved_tensors(self):
        out = self.to_save
        self.to_save = []
        return out

    @classmethod
    def apply(cls, *inputs):
        # forward
        ctx = cls() # todo...
        outs = cls.forward(ctx, *inputs)
        outList = outs if isinstance(outs, (list, tuple)) else [outs]

        # build autograd graph
        in_tensors = [(t, i) for i, t in enumerate(inputs) 
            if isinstance(t, (Tensor, spt.SparseTensor)) and t.requires_grad]
        # out_tensors all become requires_grad == True if at least one of in_tensors requires_grad == True
        out_tensors = [t for t in outList if isinstance(t, (Tensor, spt.SparseTensor))] \
            if len(in_tensors) > 0 else []
        assert len(out_tensors) == 1, 'currently only support output 1 tensor'
        out_tensor = out_tensors[0]
        # assign current function node to outputs, for further functions.
        out_tensor.grad_fn = Node(ctx, cls.backward, 0)
        # retrieve previous funtion nodes (links)
        for t, i in in_tensors:
            if not isinstance(t.grad_fn, Node):
                # leaf_tensor, grad_fn = accumulateGrad
                t.grad_fn = GradAcc(t)
            out_tensor.grad_fn.next_functions.append((t.grad_fn, i))

        return outs
        

def dfs(node: Node, vis: set, outList: list):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Node):
                for in_fn, _ in node.next_functions:
                    dfs(in_fn, vis, outList)
                outList.append(node)


def sparse_tensor_backward(tensor, grad):
    if not isinstance(tensor.grad_fn, Node):
        return
    # topological sort
    visibleSet = set()
    outList = []
    grad_fn = tensor.grad_fn
    dfs(grad_fn, visibleSet, outList)
    # backward
    visibleSet = set()
    grad_dict = {grad_fn: grad}
    for grad_fn in reversed(outList):
        in_grad_fns = grad_fn.next_functions
        in_grads = grad_fn(grad_dict[grad_fn])
        for in_grad_fn, i in in_grad_fns:
            grad = in_grads[i]
            if isinstance(grad, (Tensor, spt.SparseTensor)):
                grad_dict[in_grad_fn] = grad if in_grad_fn not in visibleSet else grad_dict[in_grad_fn] + grad
            
            visibleSet.add(in_grad_fn)
