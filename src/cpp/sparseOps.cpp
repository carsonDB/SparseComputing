#include "sparseOps.h"


/**
 * @param input e.g. [batch, channel(sparse), height, width]
 * (not considered) @param dim must dim != sparse_dim. Otherwise, just use dense way in python .
 */
vector<SparseTensor> sparse_batchMeanVar(SparseTensor& input) {
    TORCH_CHECK(input.sparse_dim() == 1);
    TORCH_CHECK(input.indices().dim() == 4 && input.values().dim() == 4);
    // todo...
    auto numel = input.indices().numel();
    auto indice_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto value_ptr = input.values().contiguous().data_ptr<float>();
    unordered_map<int64_t, normStats> out_map; // id (sparse) -> normStats
    for (const auto i : c10::irange(numel)) {
        auto id = indice_ptr[i];
        auto val = value_ptr[i];
        out_map[id].count++;
        out_map[id].sum += val;
    }
    int64_t size = out_map.size();
    auto mean = create_sparse_IdVal({size});
    int offset = 0;
    for (const auto& item : out_map) {
        mean.id_ptr[offset] = item.first;
        mean.val_ptr[offset] = item.second.sum / item.second.count;
        offset++;
    }

    auto var = create_sparse_IdVal({size});
    for (const auto i : c10::irange(numel)) {
        auto id = indice_ptr[i];
        auto val = value_ptr[i];
        auto item = out_map[id];
        item.diffSquared += pow(val - item.sum / item.count, 2);
    }

    offset = 0;
    for (const auto& item : out_map) {
        var.id_ptr[offset] = item.first;
        var.val_ptr[offset] = item.second.diffSquared / (item.second.count - 1); // todo... n-1 or n ??
        offset++;
    }

    return { 
        SparseTensor(mean.indices, mean.values, 0, input.range()),
        SparseTensor(var.indices, var.values, 0, input.range())
    };
}


/**
 * reduce axis should not include sparse_dim. Otherwise, use normal dense tensor way.
 * @param axis need to be reduced
 */
SparseTensor sparse_reduce_template(
    const SparseTensor& input, 
    const set<int64_t> axis, 
    const bool keepdim,
    const templateFn& f
) {
    TORCH_CHECK(axis.count(input.sparse_dim()) == 0, "axis should not include sparse_dim");
    TORCH_CHECK(input.indices().dim() == input.values().dim(), "only support value=Scalar yet");
    const auto in_id_ptr = input.c_indices_ptr();
    const auto in_val_ptr = input.c_values_ptr();
    const auto input_shape = input.indices().sizes().vec();

    vector<int64_t> rest_sizes; // rest sizes without sparse dim
    vector<int64_t> reduce_sizes; // reduced sizes + sparse dim
    for (auto const i : c10::irange(input_shape.size())) {
        axis.count(i) == 0 && (int)i != input.sparse_dim() ?
            rest_sizes.push_back(input_shape[i]) :
            reduce_sizes.push_back(input_shape[i]);
    }
    // ensure outside loop execute at least once.
    if (rest_sizes.size() == 0) rest_sizes.push_back(1);

    auto locals_merge = [&](
        const vector<int64_t>& rest_locals, 
        const vector<int64_t>& reduce_locals, 
        vector<int64_t>& locals_all
    ) {
        int p1 = 0, p2 = 0;
        for (const auto i : c10::irange(input_shape.size())) {
            locals_all[i] = 
                axis.count(i) == 0 && (int)i != input.sparse_dim() ? 
                    rest_locals[p1++] : 
                    reduce_locals[p2++];
        }
    };

    // get max_size
    int max_size = 0;
    sizes_for(rest_sizes, true, [&](vector<int64_t>& locals1, int64_t global) {
        unordered_set<int64_t> id_set;
        auto offsets = vector<int64_t>(input_shape.size(), 0);
        sizes_for(reduce_sizes, false, [&](vector<int64_t>& locals2, int64_t global) {
            locals_merge(locals1, locals2, offsets);
            auto offset = local2global_offset(input_shape, offsets);
            id_set.insert(in_id_ptr[offset]);
        });
        max_size = max(max_size, (int)id_set.size());
    });

    // create output with keepdim
    vector<int64_t> out_shape;
    auto out_sparse_dim = input.sparse_dim();
    for (const auto i : c10::irange(input_shape.size())) {
        if ((int)i == input.sparse_dim()) {
            out_shape.push_back(max_size);
            out_sparse_dim = out_shape.size() - 1;
            continue;
        }
        if (axis.count(i) == 0)
            out_shape.push_back(input_shape[i]);
        else if (keepdim)
            out_shape.push_back(1);
    }
    auto out = create_sparse_IdVal(out_shape);
    
    sizes_for(rest_sizes, true, [&](vector<int64_t>& locals1, int64_t global) {
        unordered_map<int64_t, int64_t> id2offset; // sparse id -> offset of sparse dim of output
        int num_item = 0; // output offset
        auto out_locals = vector<int64_t>(out_shape.size(), 0);
        auto in_offsets = vector<int64_t>(input_shape.size(), 0);
        sizes_for(reduce_sizes, false, [&](vector<int64_t>& locals2, int64_t global) {
            locals_merge(locals1, locals2, in_offsets);
            auto offset = local2global_offset(input_shape, in_offsets);
            auto id = in_id_ptr[offset];
            auto val = in_val_ptr[offset];
            if (id2offset.count(id) == 0) {
                id2offset[id] = num_item++;
            }
            // out local offsets, erase reduce axis
            for (size_t i = 0, n = 0; i < in_offsets.size(); i++) {
                if (axis.count(i) == 0)
                    out_locals[n++] = in_offsets[i];
                else if (keepdim)
                    out_locals[n++] = 0;
            }
            out_locals[out_sparse_dim] = id2offset[id];
            auto out_offset = local2global_offset(out_shape, out_locals);
            out.id_ptr[out_offset] = id;
            auto acc = out.val_ptr[out_offset];
            out.val_ptr[out_offset] = f(acc, val); // todo... extend to prod...
        });
    });
    
    return SparseTensor(out.indices, out.values, out_sparse_dim, input.range());
}


/**
 * currently, union of two inputs at sparse_dim.
 * Broadcast rules follow: https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param f (float, float) -> float, operation of each pair.
 */
SparseTensor sparse_elementwise_template(SparseTensor& input1, SparseTensor& input2, const templateFn& f) {
    TORCH_CHECK(input1.range() == input2.range());
    auto shape1 = input1.indices().sizes().vec();
    auto shape2 = input2.indices().sizes().vec();
    auto sparse_dim1 = input1.sparse_dim();
    auto sparse_dim2 = input2.sparse_dim();
    TORCH_CHECK(shape1.size() - sparse_dim1 == shape2.size() - sparse_dim2); // align from tailling
    int64_t sparse_dim_from_tailing = shape1.size() - sparse_dim1 - 1;
    auto in_id_ptr1 = input1.c_indices_ptr();
    auto in_val_ptr1 = input1.c_values_ptr();
    auto in_id_ptr2 = input2.c_indices_ptr();
    auto in_val_ptr2 = input2.c_values_ptr();
    int sparse_size1 = input1._size(sparse_dim1);
    int sparse_size2 = input2._size(sparse_dim2);
    // find out related shapes
    int out_dims = max(shape1.size(), shape2.size());
    int out_sparse_dim = out_dims - sparse_dim_from_tailing - 1;
    auto unrelated_shape = vector<int64_t>(out_dims, 0); // unrelated_shape[sparse_dim] = 1
    // tl (tail): start from end
    for (const auto tl : c10::irange(out_dims)) {
        if (tl == sparse_dim_from_tailing) {
            unrelated_shape[out_dims - tl - 1] = 1;
            continue;
        }
        // shape.size() is unsigned number, won't be negative.
        auto size1 = (int)shape1.size() >= tl + 1 ? shape1[shape1.size() - tl - 1] : 1;
        auto size2 = (int)shape2.size() >= tl + 1 ? shape2[shape2.size() - tl - 1] : 1;
        TORCH_CHECK(size1 == size2 || size1 == 1 || size2 == 1, 
            "same or either size is 1 or 0, but " + to_string(size1) + " and " + to_string(size2));
        unrelated_shape[out_dims - tl - 1] = max(size1, size2);
    }
    // find max_sparse_size
    int64_t out_sparse_size = 0;
    sizes_for(unrelated_shape, true, [&](vector<int64_t> ids, int64_t global) {
        // already ids[out_sparse_dim] = 0
        unordered_set<int64_t> id_set;
        for (const auto i : c10::irange(sparse_size1)) {
            ids[out_sparse_dim] = i;
            auto id = local2global_offset_broadcast(shape1, ids);
            id_set.insert(in_id_ptr1[id]);
        }
        for (const auto i : c10::irange(sparse_size2)) {
            ids[out_sparse_dim] = i;
            auto id = local2global_offset_broadcast(shape2, ids);
            id_set.insert(in_id_ptr2[id]);
        }
        out_sparse_size = max(out_sparse_size, (int64_t)id_set.size());
    });
    // init out tensor
    vector<int64_t> out_shape = unrelated_shape;
    out_shape[out_sparse_dim] = out_sparse_size;
    auto out = create_sparse_IdVal(out_shape);
    // elementwise operation
    sizes_for(unrelated_shape, true, [&](vector<int64_t> ids, int64_t global) {
        // already ids[out_sparse_dim] = 0
        unordered_map<int64_t, float> out_map;
        unordered_set<int64_t> out_set;
        for (const auto i : c10::irange(sparse_size1)) {
            ids[out_sparse_dim] = i;
            auto offset = local2global_offset_broadcast(shape1, ids);
            auto id = in_id_ptr1[offset];
            out_map[id] = in_val_ptr1[offset];
            out_set.insert(id);
        }
        for (const auto i : c10::irange(sparse_size2)) {
            ids[out_sparse_dim] = i;
            auto offset = local2global_offset_broadcast(shape2, ids);
            auto id = in_id_ptr2[offset];
            auto num1 = out_map[id];
            auto num2 = in_val_ptr2[offset];
            out_map[id] = f(num1, num2);
            out_set.erase(id);
        }
        // when num1 != 0, num2 == 0
        for (const auto& id : out_set) {
            out_map[id] = f(out_map[id], 0);
        }
        // write map values to out tensor
        int offset = 0;
        for (const auto& item : out_map) {
            ids[out_sparse_dim] = offset;
            auto id = local2global_offset(out_shape, ids);
            out.id_ptr[id] = item.first;
            out.val_ptr[id] = item.second;
            offset++;
        }
    });

    return SparseTensor(out.indices, out.values, out_sparse_dim, input1.range());
}


/**
 * @param input2 _indices: [sparse_dims, nse], _values: [nse, dense_dims]
 */
SparseTensor sparse_elementwise_template(
    SparseTensor& input1, 
    torch::Tensor& input2, 
    const templateFn& f,
    bool inplace
) {
    TORCH_CHECK(input2.layout() == torch::kSparse);
    TORCH_CHECK(input2._values().dim() == 1, "input2 currently not support dense tensor");
    auto indices1 = input1.indices();
    auto values1 = input1.values();
    auto id1_sizes = indices1.sizes().vec();
    auto val1_sizes = values1.sizes().vec();
    auto id1_ptr = input1.c_indices_ptr();
    auto val1_ptr = input1.c_values_ptr();
    auto indices2 = input2._indices();
    auto id2_ptr = indices2.data_ptr<int64_t>();
    auto val2_ptr = input2._values().data_ptr<float>();
    auto sparse_dim = input1.sparse_dim();
    auto sparse_dims2 = indices2.size(0);
    TORCH_CHECK(sparse_dims2 == values1.dim());
    auto sparse_dims1 = indices1.dim();
    TORCH_CHECK(input1.indices().dim() <= sparse_dims2);
    auto nse = indices2.size(1);
    auto id1_locals = vector<int64_t>(sparse_dims1, 0);
    auto val1_locals = vector<int64_t>(values1.dim(), 0);

    auto out = input1;
    if (!inplace) {
        auto ids = indices1.clone().detach();
        auto vals = values1.clone().detach();
        out = SparseTensor(ids, vals, sparse_dim, input1.range());
    }
    // auto out_id_ptr = out.c_indices_ptr();
    auto out_val_ptr = out.c_values_ptr();
    for (const auto i : c10::irange(nse)) {
        // traversal along the sparse dim, find sparse1_id == sparse2_id
        for (const auto j : c10::irange(input1._size(sparse_dim))) {
            // clip beginning part of locals of input2
            for (const auto k : c10::irange(sparse_dims1)) {
                auto id = id2_ptr[k * nse + i];
                id1_locals[k] = id;
                val1_locals[k] = id;
                TORCH_CHECK(id >= 0, "indices of input2 should >= 0.");
                TORCH_CHECK(k == sparse_dim ? id < input1.range() : id < input1._size(k),
                    "indices of input2 exceed input1 range.");
            }
            for (const auto k : c10::irange(sparse_dims1, sparse_dims2)) {
                val1_locals[k] = id2_ptr[k * nse + i];
            }
            auto sparse2_id = id1_locals[sparse_dim];
            id1_locals[sparse_dim] = j;
            val1_locals[sparse_dim] = j;
            auto offset = local2global_offset(id1_sizes, id1_locals);
            auto sparse1_id = id1_ptr[offset];
            // find the corresponding entry
            if (sparse1_id == sparse2_id) {
                auto offset = local2global_offset(val1_sizes, val1_locals);
                // out_id_ptr[offset] = sparse1_id; // out_indices same with id1_indices
                out_val_ptr[offset] = f(val1_ptr[offset], val2_ptr[i]);
            }
        }
    }

    return out;
}
