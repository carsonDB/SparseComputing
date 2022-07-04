#include "sparseTensor.h"

/**
 * Only one dimension to be sparse
 * Structure:
 *  indices: sparse_shape
 *  values: sparse_shape + dense_shape
 * @param dim sparse dim in sparse_shape
 * @param range sparse range at sparse dim
 */
void SparseTensor::build_id2offset_map() {
    id2offset_map.clear();
    auto size = _indices.numel();
    _indices = _indices.contiguous();
    auto id_ptr = _indices.data_ptr<int64_t>();
    for (const auto i : c10::irange(size)) {
        auto id = id_ptr[i];
        if (id2offset_map.find(id) != id2offset_map.end())
            id2offset_map[id].push_back(i);
        else
            id2offset_map[id] = {i};
    }
}


/**
 * First sort along sparse_dim, then unique. Unset entries is all zeros.
 */
void SparseTensor::coalesce() {
    auto& ids = _indices;
    auto& vals = _values;
    auto sparse_size = ids.size(_sparse_dim);
    auto unrelated_sizes = vector<int>(ids.dim() - 1, 0);
    // unrelated_sizes...

    // sort
    auto tuple = ids.sort(_sparse_dim);
    auto& sorted_ids = tuple.get<0>;
    auto& ids_map = tuple.get<1>;

    // unique
    auto unique_ids = torch::zeros_like(ids);
    auto unique_vals = torch::zeros_like(vals);
    auto valid_length = torch::zeros(unrelated_sizes, torch::dtype(torch::kI64));
    sizes_for(unrelated_sizes, true, [&](int64_t locals, int64_t global) {

    });

    // remove all same zeros tailing
    auto max_len = valid_length.max();
    auto out_ids = unique_ids.index(...);
    auto out_vals = unique_vals.index(...);

    return update_with(out_ids, out_vals, _sparse_dim);
}