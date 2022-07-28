#ifndef SPARSE_MM_H
#define SPARSE_MM_H

#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>

#include "sparseTensor.h"
#include "utils.h"

using namespace std;


torch::Tensor sparseLinear_forward(SparseTensor& input, torch::Tensor& weight);
SparseTensor sparseLinear_backward(SparseTensor& input, torch::Tensor& grad);
torch::Tensor linear_backward_coo_v0(SparseTensor& input, torch::Tensor& grad);

template<typename T> torch::Tensor linear_backward_coo_template(SparseTensor& input, torch::Tensor& grad);
inline torch::Tensor linear_backward_coo(SparseTensor& input, torch::Tensor& grad) {
    return (input.dtype() == torch::kF32 && grad.dtype() == torch::kF32) ?
        linear_backward_coo_template<float>(input, grad) :
        linear_backward_coo_template<double>(input, grad);
}

#endif