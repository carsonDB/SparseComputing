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

public:
    SparseTensor(torch::Tensor& indices, torch::Tensor& values, int64_t sparse_dim, int64_t range) {
        // check indices is prefix of values
        TORCH_CHECK(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < range);
        TORCH_CHECK(sparse_dim < indices.dim(), "sparse_dim exceeds indices.dim()");
        _indices = indices;
        _values = values;
        _sparse_dim = sparse_dim;
        _range = range;
    }
    void coalesce();
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
    const int64_t sparse_dim() const { return _sparse_dim; }
    const int64_t range() const { return _range; }
    SparseTensor update_with(torch::Tensor& indices, torch::Tensor& values, int64_t sparse_dim) {
        auto out = SparseTensor(indices, values, sparse_dim, _range);
        if (torch::equal(indices, _indices) && sparse_dim == _sparse_dim)
            out.c_set_id2offset_map(id2offset_map);
        return out;
    }
    
    vector<int64_t>& c_id2offset(int64_t in) { return id2offset_map[in]; }
    void c_set_id2offset_map(Id2OffsetMap map) { id2offset_map = map; }
    int64_t* c_indices_ptr() const { return _indices.contiguous().data_ptr<int64_t>(); }
    float* c_values_ptr() const { return _values.contiguous().data_ptr<float>(); }
};

#endif