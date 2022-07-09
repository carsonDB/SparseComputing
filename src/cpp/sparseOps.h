#ifndef SPARSE_OPS_H
#define SPARSE_OPS_H

#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "sparseTensor.h"
#include "utils.h"

using namespace std;

template<typename T>
using templateFn = function<T(T, T)>;

struct normStats {
    float sum;
    int count;
    float diffSquared;
};

vector<SparseTensor> sparse_batchMeanVar(SparseTensor& input);

template<typename T>
SparseTensor sparse_reduce_template(
    const SparseTensor& input, 
    const set<int64_t> axis, 
    const bool keepdim,
    const templateFn<T>& f
);

inline SparseTensor reduce_sum(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    return (input.dtype() == torch::kF32) ?
        sparse_reduce_template<float>(input, axis, keepdim, [](auto acc, auto x) { return acc + x; }) :
        sparse_reduce_template<double>(input, axis, keepdim, [](auto acc, auto x) { return acc + x; });
}

inline SparseTensor reduce_count_nonzero(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    return (input.dtype() == torch::kF32) ? 
        sparse_reduce_template<float>(input, axis, keepdim, [](auto acc, auto x) { return acc + float(x != 0); }) :
        sparse_reduce_template<double>(input, axis, keepdim, [](auto acc, auto x) { return acc + double(x != 0); });
}

inline SparseTensor reduce_prod(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    TORCH_CHECK(false, "currently need to solve init values of output problem.");
    return (input.dtype() == torch::kF32) ? 
        sparse_reduce_template<float>(input, axis, keepdim, [](auto acc, auto x) { return acc * x; }) :
        sparse_reduce_template<double>(input, axis, keepdim, [](auto acc, auto x) { return acc * x; });
}



template<typename T>
SparseTensor sparse_elementwise_template(SparseTensor& input1, SparseTensor& input2, const templateFn<T>& f);
template<typename T>
SparseTensor sparse_elementwise_template(SparseTensor& input1, torch::Tensor& input2, const templateFn<T>& f, bool inplace);


inline SparseTensor elementwise_add(SparseTensor& input1, SparseTensor& input2) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a + b; }) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a + b; });
}

inline SparseTensor elementwise_add(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a + b; }, inplace) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a + b; }, inplace);
}

inline SparseTensor elementwise_sub(SparseTensor& input1, SparseTensor& input2) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a - b; }) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a - b; });
}

inline SparseTensor elementwise_sub(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a - b; }, inplace) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a - b; }, inplace);
}

inline SparseTensor elementwise_mul(SparseTensor& input1, SparseTensor& input2) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a * b; }) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a * b; });
}

inline SparseTensor elementwise_mul(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a * b; }, inplace) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a * b; }, inplace);
}

inline SparseTensor elementwise_div(SparseTensor& input1, SparseTensor& input2) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a / b; }) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a / b; });
}

inline SparseTensor elementwise_div(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return (input1.dtype() == torch::kF32 && input2.dtype() == torch::kF32) ?
        sparse_elementwise_template<float>(input1, input2, [](auto a, auto b) { return a / b; }, inplace) :
        sparse_elementwise_template<double>(input1, input2, [](auto a, auto b) { return a / b; }, inplace);
}


#endif