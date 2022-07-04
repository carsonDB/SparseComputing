import math
import time
import torch

from src.python.sparseTypes import SparseTensor


min_t = math.inf
sum_t = 0

runs = 100
for _ in range(runs):
    # init
    input = SparseTensor.from_dense(torch.randn(64, 128, 8, 8), 32, 1)
    dim = (0,2,3)
    eps = 1e-5

    start = time.time()
    
    count = input.count_nonzero(dim, True) # 25 ms
    mean = input.sum(dim, True) / count # 24 ms
    xmu = input - mean # 71 ms
    sq = xmu ** 2 # 0.5 ms
    var = sq.sum(dim, True) / count # 96 ms
    sqrtvar = (var + eps).sqrt() # 0.2 ms
    output = xmu / sqrtvar # 102 ms

    elapsed = time.time() - start
    min_t = min(min_t, elapsed)
    sum_t += elapsed


scale = 1000 # ms
min_t *= scale
avg_t = sum_t / runs * scale

print('Elapsed: min({0:.3f})ms, avg({1:.3f})ms'.format(min_t, avg_t))
