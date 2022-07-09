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
template<typename T>
SparseTensor coalesce_template(SparseTensor& input) {
    auto ids = input.indices();
    auto vals = input.values();
    TORCH_CHECK(ids.dim() == vals.dim(), "currently value should be scalar.");
    auto sparse_dim = input.sparse_dim();
    auto sizes = ids.sizes().vec();
    auto val_ptr = vals.contiguous().data_ptr<T>();
    auto sparse_size = sizes[sparse_dim];
    // sparse size is 0 or 1
    if (sparse_size <= 1) return input.update_with(ids, vals, sparse_dim);
    // sparse_stride: distance between nearest sparse id
    int sparse_stride = 1;
    for (size_t i = sparse_dim + 1; i < sizes.size(); i++)
        sparse_stride *= sizes[i];
    // unrelated_sizes: only sparse_size = 1
    vector<int64_t> unrelated_sizes = sizes;
    unrelated_sizes[sparse_dim] = 1;
    int64_t unrelated_numel = 1;
    for (const auto i : unrelated_sizes)
        unrelated_numel *= i;
    // sort
    auto tuple = ids.sort(sparse_dim);
    auto sorted_id_ptr = get<0>(tuple).data_ptr<int64_t>();
    auto sorted_map_ptr = get<1>(tuple).data_ptr<int64_t>();
    // unique
    auto unique_ids = torch::zeros_like(ids);
    auto unique_id_ptr = unique_ids.data_ptr<int64_t>();
    auto unique_vals = torch::zeros_like(vals);
    auto unique_val_ptr = unique_vals.data_ptr<T>();
    auto valid_length = vector<int64_t>(unrelated_numel, 0);
    sizes_for(unrelated_sizes, true, [&](vector<int64_t>& locals, auto global) {
        auto start = local2global_offset(sizes, locals);
        auto end = start + sparse_size * sparse_stride;
        // unique
        int j = start; // unique offset
        for (int i = start; i < end - sparse_stride; i += sparse_stride) {
            unique_val_ptr[j] += val_ptr[start + sorted_map_ptr[i] * sparse_stride];
            if (sorted_id_ptr[i] != sorted_id_ptr[i + sparse_stride]) {
                unique_id_ptr[j] = sorted_id_ptr[i];
                j += sparse_stride;
            }
        }
        // Store the last element
        unique_id_ptr[j] = sorted_id_ptr[end - sparse_stride];
        unique_val_ptr[j] += val_ptr[start + sorted_map_ptr[end - sparse_stride] * sparse_stride];
        valid_length[global] = (j - start) / sparse_stride + 1;
    });
    // remove all same zeros tailing
    auto max_len = *max_element(valid_length.begin(), valid_length.end());
    vector<at::indexing::TensorIndex> slices(sizes.size(), "...");
    slices[sparse_dim] = torch::indexing::Slice(0, max_len, 1);
    auto out_ids = unique_ids.index(slices);
    auto out_vals = unique_vals.index(slices);

    return input.update_with(out_ids, out_vals, sparse_dim);
}

template SparseTensor coalesce_template<float>(SparseTensor& input);
template SparseTensor coalesce_template<double>(SparseTensor& input);


