#ifndef SPARSE_CONV_H
#define SPARSE_CONV_H

#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "sparseTensor.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;


struct size_hw {
    int64_t h;
    int64_t w;
    size_hw(vector<int64_t>& vec): h(vec[0]), w(vec[1]) {
        TORCH_CHECK(vec.size() == 2, "size_hw only accpet length 2");
    }
};

template<typename T>
torch::Tensor unfold(const torch::Tensor& input, size_hw kernel, size_hw padding, size_hw stride, size_hw dilation);

vector<SparseTensor> sparseConv2d_forward(
    SparseTensor& input,
    SparseTensor& weight,
    vector<int64_t> stride,
    vector<int64_t> padding
);

torch::Tensor sparseConv2d_backward(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    vector<int64_t>& kernel
);

#endif