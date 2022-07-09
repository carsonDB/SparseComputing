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

template<typename T>
vector<SparseTensor> sparseConv2d_forward_template(
    SparseTensor& input,
    SparseTensor& weight,
    vector<int64_t> stride,
    vector<int64_t> padding
);


inline vector<SparseTensor> sparseConv2d_forward(
    SparseTensor& input,
    SparseTensor& weight,
    vector<int64_t> stride,
    vector<int64_t> padding
) {
    if (input.dtype() == torch::kF32 && weight.dtype() == torch::kF32)
        return sparseConv2d_forward_template<float>(input, weight, stride, padding);
    else
        return sparseConv2d_forward_template<double>(input, weight, stride, padding);
};


template<typename T>
torch::Tensor sparseConv2d_backward_template(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    vector<int64_t>& kernel
);

inline torch::Tensor sparseConv2d_backward(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs, 
    vector<int64_t>& kernel
) {
    if (grad.dtype() == torch::kF32 && unfolded_inputs.dtype() == torch::kF32)
        return sparseConv2d_backward_template<float>(grad, unfolded_inputs, kernel);
    else
        return sparseConv2d_backward_template<double>(grad, unfolded_inputs, kernel);
}



#endif