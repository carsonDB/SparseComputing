#ifndef SPARSE_OPS_H
#define SPARSE_OPS_H

#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "sparseTensor.h"
#include "utils.h"

using namespace std;

using templateFn = function<float(float, float)>;

struct normStats {
    float sum;
    int count;
    float diffSquared;
};

vector<SparseTensor> sparse_batchMeanVar(SparseTensor& input);

SparseTensor sparse_reduce_template(
    const SparseTensor& input, 
    const set<int64_t> axis, 
    const bool keepdim,
    const templateFn& f
);

inline SparseTensor reduce_sum(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    return sparse_reduce_template(input, axis, keepdim, [](float acc, float x) { return acc + x; });
}

inline SparseTensor reduce_count_nonzero(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    return sparse_reduce_template(input, axis, keepdim, [](float acc, float x) { 
        return acc + float(x != 0); 
    });
}

inline SparseTensor reduce_prod(SparseTensor& input, set<int64_t> axis, bool keepdim) {
    TORCH_CHECK(false, "currently need to solve init values of output problem.")
    return sparse_reduce_template(input, axis, keepdim, [](float acc, float x) { return acc * x; });
}

SparseTensor sparse_elementwise_template(SparseTensor& input1, SparseTensor& input2, const templateFn& f);
SparseTensor sparse_elementwise_template(SparseTensor& input1, torch::Tensor& input2, const templateFn& f, bool inplace);


inline SparseTensor elementwise_add(SparseTensor& input1, SparseTensor& input2) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a + b; });
}

inline SparseTensor elementwise_add(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a + b; }, inplace);
}

inline SparseTensor elementwise_sub(SparseTensor& input1, SparseTensor& input2) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a - b; });
}

inline SparseTensor elementwise_sub(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a - b; }, inplace);
}

inline SparseTensor elementwise_mul(SparseTensor& input1, SparseTensor& input2) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a * b; });
}

inline SparseTensor elementwise_mul(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a * b; }, inplace);
}

inline SparseTensor elementwise_div(SparseTensor& input1, SparseTensor& input2) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a / b; });
}

inline SparseTensor elementwise_div(SparseTensor& input1, torch::Tensor& input2, bool inplace) {
    return sparse_elementwise_template(input1, input2, [](float a, float b) { return a / b; }, inplace);
}


#endif