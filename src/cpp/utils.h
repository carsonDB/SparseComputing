#ifndef UTILS_H
#define UTILS_H
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>
#include <string>

namespace thIndex = torch::indexing;
using namespace std;


inline void check_dim(torch::Tensor& x, int dim) {
    TORCH_CHECK(x.dim() == dim, "Tensor must be dim = ", to_string(dim), " but ", x.dim());
}

struct OutSparseTensor {
    torch::Tensor indices;
    torch::Tensor values;
    int64_t* id_ptr;
    float* val_ptr;
};

inline OutSparseTensor create_sparse_IdVal(const vector<int64_t>& id_shape, const vector<int64_t>& val_shape) {
    auto indices = torch::zeros(id_shape, torch::dtype(torch::kI64));
    auto values = torch::zeros(val_shape, torch::dtype(torch::kF32));
    auto id_ptr = indices.data_ptr<int64_t>();
    auto val_ptr = values.data_ptr<float>();

    return {indices, values, id_ptr, val_ptr};
}

inline OutSparseTensor create_sparse_IdVal(const vector<int64_t>& shape) {
    return create_sparse_IdVal(shape, shape);
}


inline int64_t local2global_offset(const vector<int64_t>& sizes, const vector<int64_t>& locals) {
    TORCH_CHECK(sizes.size() == locals.size());

    auto dims = sizes.size();
    int64_t offset = 0;
    for (const auto i : c10::irange(dims)) {
        offset += locals[i];
        if (i + 1 < dims)
            offset *= sizes[i+1];
    }
    return offset;
}

/**
 * @param sizes should be value copied, since it will be modified.
 * Each id in locals will remainder.
 * If locals is longer than sizes, then prefix sizes with 1.
 */
inline int64_t local2global_offset_broadcast(vector<int64_t> sizes, const vector<int64_t>& locals) {
    TORCH_CHECK(sizes.size() <= locals.size());
    size_t num_prefix = locals.size() - sizes.size();
    sizes.insert(sizes.begin(), num_prefix, 1);

    auto dims = sizes.size();
    int64_t offset = 0;
    for (const auto i : c10::irange(dims)) {
        offset += locals[i] % sizes[i];
        if (i + 1 < dims)
            offset *= sizes[i+1];
    }
    return offset;
}

inline void global2local_offset(
    const vector<int64_t>& sizes, 
    int64_t offset, 
    vector<int64_t>& out_vec
) {
    auto dims = sizes.size();
    for (const auto j : c10::irange(dims)) {
        auto dim = dims - j - 1;
        out_vec[dim] = offset % sizes[dim];
        offset /= sizes[dim];
    }
}

/**
 * parallel iteration
 * @param sizes vector of shape;
 * @tparam f (vector<int64_t> local_offsets, int64_t global_offset) -> void 
 */
inline void sizes_for(
    vector<int64_t>& sizes,
    bool parallel,
    const function<void (vector<int64_t>&, int64_t)>& f
) {
    int numel = 1;
    for (const auto s : sizes) 
        numel *= s;
    auto grain_size = parallel ? 0 : numel;
    at::parallel_for(0, numel, grain_size, [&](int64_t start, int64_t end) {
        vector<int64_t> ids(sizes.size(), 0);
        for (const auto i : c10::irange(start, end)) {
            global2local_offset(sizes, i, ids);
            f(ids, i);
        }
    });
}


torch::Tensor reservoir_sampling(int n, int k);

#endif