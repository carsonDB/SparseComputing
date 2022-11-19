#ifndef UTILS_H
#define UTILS_H
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>
#include <string>
#include <limits>

namespace thIndex = torch::indexing;
using namespace std;


template<typename T>
struct OutSparseTensor_T {
    torch::Tensor indices;
    torch::Tensor values;
    int64_t* id_ptr;
    T* val_ptr;
};

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

template<typename T>
inline OutSparseTensor_T<T> create_sparse_IdVal_options(
    const vector<int64_t>& id_shape, 
    const vector<int64_t>& val_shape,
    const torch::TensorOptions& options
) {
    auto indices = torch::zeros(id_shape, options.dtype(torch::kI64));
    auto values = torch::zeros(val_shape, options);
    auto id_ptr = indices.data_ptr<int64_t>();
    auto val_ptr = values.data_ptr<T>();

    return {indices, values, id_ptr, val_ptr};
}

inline OutSparseTensor create_sparse_IdVal(const vector<int64_t>& shape) {
    return create_sparse_IdVal(shape, shape);
}

template<typename T>
inline OutSparseTensor_T<T> create_sparse_IdVal_options(
    const vector<int64_t>& shape, 
    const torch::TensorOptions& options
) {
    return create_sparse_IdVal_options<T>(shape, shape, options);
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
 * @param locals must have same length with params `sizes`
 */
inline int64_t local2global_offset(const vector<int64_t>& sizes, const int64_t* locals) {
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

/**
 * @param sizes should be value copied, since it will be modified.
 * @param local_size length of locals array.
 * Each id in locals will remainder.
 * If locals is longer than sizes, then prefix sizes with 1.
 */
inline int64_t local2global_offset_broadcast(
    vector<int64_t> sizes, 
    const int64_t* locals, 
    size_t local_size
) {
    TORCH_CHECK(sizes.size() <= local_size);
    size_t num_prefix = local_size - sizes.size();
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

inline void global2locals_offset(
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
 * @param out_ptr array (len = sizes.size())
 */
inline void global2locals_offset(
    const vector<int64_t>& sizes, 
    int64_t offset,
    int64_t* out_ptr
) {
    int dims = sizes.size();
    for (const auto j : c10::irange(dims)) {
        auto dim = dims - j - 1;
        out_ptr[dim] = offset % sizes[dim];
        offset /= sizes[dim];
    }
}

inline void parallel_for(int64_t size, bool parallel, const function<void(int64_t)>& f) {
    auto grain_size = parallel ? 0 : size;
    at::parallel_for(0, size, grain_size, [&](int64_t start, int64_t end) {
        for (const auto& i : c10::irange(start, end)) {
            f(i);
        }
    });
}


/**
 * parallel iteration
 * @param sizes vector of shape;
 * @tparam f (vector<int64_t> local_offsets, int64_t global_offset) -> void 
 */
inline void sizes_for_vec(
    vector<int64_t>& sizes,
    bool parallel,
    const function<void (vector<int64_t>, int64_t)>& f
) {
    int numel = accumulate(sizes.begin(), sizes.end(), 1, multiplies<int64_t>());
    // for (const auto s : sizes) 
    //     numel *= s;
    auto grain_size = parallel ? 0 : numel;
    at::parallel_for(0, numel, grain_size, [&](int64_t start, int64_t end) {
        vector<int64_t> ids(sizes.size(), 0);
        for (const auto i : c10::irange(start, end)) {
            global2locals_offset(sizes, i, ids);
            f(ids, i);
        }
    });
}


/**
 * parallel iteration
 * @param sizes vector of shape;
 * @tparam f (int64_t* local_offsets, int64_t global_offset) -> void 
 */
inline void sizes_for(
    vector<int64_t>& sizes,
    bool parallel,
    const function<void (int64_t*, int64_t)>& f
) {
    int numel = accumulate(sizes.begin(), sizes.end(), 1, multiplies<int64_t>());
    // for (const auto s : sizes) 
    //     numel *= s;
    auto grain_size = parallel ? 0 : numel;
    at::parallel_for(0, numel, grain_size, [&](int64_t start, int64_t end) {
        // vector<int64_t> ids(sizes.size(), 0);
        int64_t ids[sizes.size()];
        for (const auto i : c10::irange(start, end)) {
            global2locals_offset(sizes, i, ids);
            f(ids, i);
        }
    });
}


torch::Tensor reservoir_sampling(int n, int k);


pair<torch::Tensor, torch::Tensor> sorted_merge(const torch::Tensor& input, set<int64_t>& axis, int64_t sorted_axis);



#endif