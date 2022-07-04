from typing import Optional, List, Tuple, Union
import collections
from itertools import repeat


size_2_t = Union[int, Tuple[int, int]]
size_n_t = Union[int, Tuple[int, ...]]
size_n_t_none = Union[None, int, Tuple[int, ...]]

def norm_tuple(x: Union[int, Tuple], n: int):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
