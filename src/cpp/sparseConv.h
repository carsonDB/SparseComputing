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
    size_hw(pair<int64_t, int64_t>& pair): h(pair.first), w(pair.second) {}
};

template<typename T>
torch::Tensor unfold_template(
    const torch::Tensor& input, 
    size_hw kernel, 
    size_hw padding, 
    size_hw stride, 
    size_hw dilation
);

using int2 = pair<int64_t, int64_t>;
inline torch::Tensor unfold(const torch::Tensor& input, int2 kernel, int2 padding, int2 stride, int2 dilation) {
    if (input.dtype() == torch::kF32)
        return unfold_template<float>(input, kernel, padding, stride, dilation);
    else if (input.dtype() == torch::kF64)
        return unfold_template<double>(input, kernel, padding, stride, dilation);
    else if (input.dtype() == torch::kI64)
        return unfold_template<int64_t>(input, kernel, padding, stride, dilation);
    else
        TORCH_CHECK(false, "no support input.dtype()");
}

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
torch::Tensor sparseConv2d_backward_template_v0(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    vector<int64_t>& kernel
);

inline torch::Tensor sparseConv2d_backward_v0(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs, 
    vector<int64_t>& kernel
) {
    if (grad.dtype() == torch::kF32 && unfolded_inputs.dtype() == torch::kF32)
        return sparseConv2d_backward_template_v0<float>(grad, unfolded_inputs, kernel);
    else
        return sparseConv2d_backward_template_v0<double>(grad, unfolded_inputs, kernel);
}

using SparseWeightGrads = pair<torch::Tensor, SparseTensor>;

template<typename T>
SparseWeightGrads conv2d_backward_template(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    SparseTensor& weight,
    vector<int64_t>& kernel
);

inline SparseWeightGrads conv2d_backward(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    SparseTensor& weight,
    vector<int64_t>& kernel
) {
    if (grad.dtype() == torch::kF32 && unfolded_inputs.dtype() == torch::kF32)
        return conv2d_backward_template<float>(grad, unfolded_inputs, weight, kernel);
    else
        return conv2d_backward_template<double>(grad, unfolded_inputs, weight, kernel);
}

#endif