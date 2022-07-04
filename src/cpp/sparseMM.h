#ifndef SPARSE_MM_H
#define SPARSE_MM_H

#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>

#include "sparseTensor.h"
#include "utils.h"

using namespace std;


torch::Tensor sparseDenseMM_forward(SparseTensor& input, torch::Tensor& weight);
SparseTensor sparseDenseMM_backward(SparseTensor& input, torch::Tensor& grad);
torch::Tensor sparseDenseMM_backward_coo(SparseTensor& input, torch::Tensor& grad);


#endif