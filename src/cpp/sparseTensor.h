#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include <torch/extension.h>
#include <vector>

#include "utils.h"

using namespace std;


using Id2OffsetMap = unordered_map<int64_t, vector<int64_t>>;

class SparseTensor {
    torch::Tensor _indices;
    torch::Tensor _values;
    int64_t _range;
    int64_t _sparse_dim;
    // id (sparse) -> global offset of all entries
    void build_id2offset_map();
    Id2OffsetMap id2offset_map;
    bool _is_coalesced = false;
    torch::Tensor coalesced_map;
public:
    SparseTensor(torch::Tensor& indices, torch::Tensor& values, int64_t sparse_dim, int64_t range) {
        // check indices is prefix of values
        auto size_eq = true;
        auto id_sizes = indices.sizes().vec();
        auto val_sizes = values.sizes().vec();
        for (size_t i = 0; i < id_sizes.size(); i++) {
            if (id_sizes[i] != val_sizes[i])
                size_eq = false;
        }
        TORCH_CHECK(size_eq, "indices is not the same size with values (prefix).");
        
        TORCH_CHECK(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < range,
            "indices: [" + to_string(indices.min().item<int64_t>()) + ", " +
            to_string(indices.max().item<int64_t>()) + "], not in range [0, " + to_string(range) + "]"
            ", when constructing SparseTensor.");
        TORCH_CHECK(sparse_dim < indices.dim(), "sparse_dim exceeds indices.dim()");
        TORCH_CHECK(values.dtype() == torch::kF32 || values.dtype() == torch::kF64, 
                "scalar type must either be float or double");
        _indices = indices;
        _values = values;
        _sparse_dim = sparse_dim;
        _range = range;
    }
    auto dtype() { return _values.dtype(); }
    const bool is_coalesced() const { return _is_coalesced; }
    SparseTensor coalesce();
    void init_rand_indices() {
        // TORCH_CHECK(!replace, "replace=true not implemented.");
        TORCH_CHECK(_indices.dim() == 2, "only support dim=2.");
        for (int i = 0; i < _indices.size(0); i++) {
            _indices.index_put_({i}, reservoir_sampling(_range, _indices.size(1))); // todo...
        }
        // update caches, since indices have changed
        build_id2offset_map();
    }
    int64_t id_count() {
        unordered_set<int64_t> id_set; // todo... cache...
        auto ptr = _indices.contiguous().data_ptr<int64_t>();
        for (const auto i : c10::irange(_indices.numel())) {
            id_set.insert(ptr[i]);
        }
        return id_set.size();
    }
    const torch::Tensor& indices() const { return _indices; }
    const torch::Tensor& values() const { return _values; }
    // size[sparse_dim] != range(), that is `values.size(...)`
    const int64_t _size(size_t dim) const { return _values.size(dim); }
    const vector<int64_t> sizes() const { 
        vector<int64_t> sizes = _values.sizes().vec();
        sizes[_sparse_dim] = _range;
        return sizes;
    }
    const int64_t sparse_dim() const { return _sparse_dim; }
    const int64_t range() const { return _range; }
    SparseTensor update_with(torch::Tensor& indices, torch::Tensor& values, int64_t sparse_dim) {
        auto out = SparseTensor(indices, values, sparse_dim, _range);
        if (torch::equal(indices, _indices) && sparse_dim == _sparse_dim) {
            out.c_set_id2offset_map(id2offset_map);
            out.c_set_coalesced(is_coalesced());
        }
        return out;
    }

// c++ only
    vector<int64_t>& c_id2offset(int64_t in) { return id2offset_map[in]; }
    void c_set_coalesced(bool is_coalesced) { 
        TORCH_CHECK(_size(sparse_dim()) <= range(), 
            "not coalesced, sparse_size (", to_string(_size(sparse_dim())), ") > id range (", to_string(range()), ")");
        _is_coalesced = is_coalesced; 
    }
    void c_set_id2offset_map(Id2OffsetMap map) { id2offset_map = map; }
};


template<typename T>
SparseTensor coalesce_template(SparseTensor& input);

inline SparseTensor SparseTensor::coalesce() {
    if (is_coalesced()) return *this;
    return (dtype() == torch::kF32) ?
        coalesce_template<float>(*this) :
        coalesce_template<double>(*this);
}


inline int64_t* indices_ptr(const SparseTensor& input) {
    TORCH_CHECK(input.indices().is_contiguous(), "SparseTensor indices must be contiguous.");
    return input.indices().data_ptr<int64_t>(); 
}

template<typename T>    
inline T* values_ptr(const SparseTensor& input) { 
    TORCH_CHECK(input.values().is_contiguous(), "SparseTensor values must be contiguous.");
    return input.values().data_ptr<T>();
}

inline int64_t* indices_ptr(const torch::Tensor& input) {
    TORCH_CHECK(input.layout() == torch::kSparse);
    TORCH_CHECK(input._indices().is_contiguous(), "input._indices() must be contiguous.");
    return input._indices().data_ptr<int64_t>(); 
}

template<typename T>
inline T* values_ptr(const torch::Tensor& input) { 
    TORCH_CHECK(input.layout() == torch::kSparse);
    TORCH_CHECK(input._values().is_contiguous(), "input._values() must be contiguous.");
    return input._values().data_ptr<T>();
}

template<typename T>
inline T* data_ptr(const torch::Tensor& input) { 
    TORCH_CHECK(input.layout() == torch::kStrided);
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
    return input.data_ptr<T>();
}


#endif