#include "sparseOps.h"


/** Deprecated
 * @param input e.g. [batch, channel(sparse), height, width]
 * (not considered) @param dim must dim != sparse_dim. Otherwise, just use dense way in python .
 */
vector<SparseTensor> sparse_batchMeanVar(SparseTensor& input) {
    TORCH_CHECK(input.sparse_dim() == 1);
    TORCH_CHECK(input.indices().dim() == 4 && input.values().dim() == 4);
    // todo...
    auto numel = input.indices().numel();
    auto indice_ptr = indices_ptr(input);
    auto value_ptr = values_ptr<float>(input);
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
template<typename T>
SparseTensor reduce_template_v0(
    const SparseTensor& input, 
    const set<int64_t> axis, 
    const bool keepdim,
    const templateFn<T>& f
) {
    TORCH_CHECK(axis.count(input.sparse_dim()) == 0, "axis should not include sparse_dim");
    TORCH_CHECK(input.indices().dim() == input.values().dim(), "only support value=Scalar yet");
    const auto in_id_ptr = indices_ptr(input);
    const auto in_val_ptr = values_ptr<T>(input);
    const auto input_shape = input.indices().sizes().vec();

    vector<int64_t> rest_sizes; // rest sizes without sparse dim
    vector<int64_t> reduce_sizes; // reduced sizes + sparse dim
    for (const int i : c10::irange(input_shape.size())) {
        axis.count(i) == 0 && i != input.sparse_dim() ?
            rest_sizes.push_back(input_shape[i]) :
            reduce_sizes.push_back(input_shape[i]);
    }
    // ensure outside loop execute at least once.
    if (rest_sizes.size() == 0) rest_sizes.push_back(1);

    auto locals_merge = [&](
        const int64_t* rest_locals, 
        const int64_t* reduce_locals, 
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
    sizes_for(rest_sizes, true, [&](auto locals1, int64_t global) {
        unordered_set<int64_t> id_set;
        auto offsets = vector<int64_t>(input_shape.size(), 0);
        sizes_for(reduce_sizes, false, [&](auto locals2, int64_t global) {
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
    auto out = create_sparse_IdVal_options<T>(out_shape, input.values().options());
    
    sizes_for(rest_sizes, true, [&](auto locals1, int64_t global) {
        unordered_map<int64_t, int64_t> id2offset; // sparse id -> offset of sparse dim of output
        int num_item = 0; // output offset
        auto out_locals = vector<int64_t>(out_shape.size(), 0);
        auto in_offsets = vector<int64_t>(input_shape.size(), 0);
        sizes_for(reduce_sizes, false, [&](auto locals2, int64_t global) {
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

template SparseTensor reduce_template_v0<float>(
    const SparseTensor& input, const set<int64_t> axis, const bool keepdim, const templateFn<float>& f);
template SparseTensor reduce_template_v0<double>(
    const SparseTensor& input, const set<int64_t> axis, const bool keepdim, const templateFn<double>& f);


/**
 * reduce axis should not include sparse_dim. Otherwise, use normal dense tensor way.
 * @param axis need to be reduced
 */
template<typename T>
SparseTensor reduce_template(
    const SparseTensor& input, 
    const set<int64_t>& axis, 
    const bool keepdim,
    const templateFn<T>& f
) {
    TORCH_CHECK(input.is_coalesced(), "input should coalesce before");
    TORCH_CHECK(axis.count(input.sparse_dim()) == 0, "axis should not include sparse_dim");
    TORCH_CHECK(input.indices().dim() == input.values().dim(), "only support value=Scalar yet");
    set<int64_t> merge_axis = axis;
    merge_axis.insert(input.sparse_dim());
    auto in_ids = input.indices();
    auto merge_pair = sorted_merge(in_ids, merge_axis, input.sparse_dim());
    auto merged_ids = merge_pair.first, merged_map = merge_pair.second;
    auto merged_ids_ptr = data_ptr<int64_t>(merged_ids), merged_map_ptr = data_ptr<int64_t>(merged_map);
    auto merged_shape = merged_ids.sizes().vec();
    // create output tensor
    auto out_shape = merged_shape;
    auto out = create_sparse_IdVal_options<T>(out_shape, input.values().options());

    // traversal each merge case
    vector<int64_t> unrelated_shape = merged_shape;
    unrelated_shape[input.sparse_dim()] = 1;
    auto merged_size = merged_shape[input.sparse_dim()];
    auto max_counts = torch::zeros(unrelated_shape, torch::kI64);
    auto max_counts_ptr = data_ptr<int64_t>(max_counts);
    auto stride = accumulate(
        merged_shape.begin() + input.sparse_dim() + 1, merged_shape.end(), 1, multiplies<int64_t>());
    auto in_vals_ptr = values_ptr<T>(input);
    sizes_for(unrelated_shape, true, [&](auto locals, auto global) {
        auto start = local2global_offset(merged_shape, locals);
        auto end = start + merged_size * stride;
        auto n = start; // merged_size == out_shape
        for (int i = start; i < end; i += stride) {
            out.val_ptr[n] = f(out.val_ptr[n], in_vals_ptr[merged_map_ptr[i]]);
            if (i + stride == end || merged_ids_ptr[i] != merged_ids_ptr[i + stride])
                out.id_ptr[n] = merged_ids_ptr[i];
            // next group of same ids
            if (i + stride < end) {
                if (merged_ids_ptr[i] < merged_ids_ptr[i + stride]) 
                    n += stride;
                else if (merged_ids_ptr[i] > merged_ids_ptr[i + stride]) 
                    break;
            }
        }
        max_counts_ptr[global] = (n - start) / stride + 1;
    });

    // slice along sparse_dim(), remove tailing 0s.
    auto max_len = max_counts.max().item<int64_t>();
    out.indices = out.indices.narrow(input.sparse_dim(), 0, max_len);
    out.values = out.values.narrow(input.sparse_dim(), 0, max_len);
    // remove dim=1 if keepdim == false
    auto out_sparse_dim = input.sparse_dim();
    if (!keepdim) {
        // latter i will not affect previous i
        for (int i = (int)out_shape.size() - 1; i >= 0; i--) {
            out.indices = out.indices.squeeze(i);
            out.values = out.values.squeeze(i);
            if (axis.count(i) > 0 && i < input.sparse_dim())
                out_sparse_dim -= 1;
        }
    }

    out.indices = out.indices.contiguous();
    out.values = out.values.contiguous();
    auto out_tensor = SparseTensor(out.indices, out.values, out_sparse_dim, input.range());
    out_tensor.c_set_coalesced(true);
    return out_tensor;
}

template SparseTensor reduce_template<float>(
    const SparseTensor& input, const set<int64_t>& axis, const bool keepdim, const templateFn<float>& f);
template SparseTensor reduce_template<double>(
    const SparseTensor& input, const set<int64_t>& axis, const bool keepdim, const templateFn<double>& f);



/**
 * currently, union of two inputs at sparse_dim.
 * Broadcast rules follow: https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param f (float/double, float/double) -> float/double, operation of each pair.
 */
template<typename T>
SparseTensor elementwise_template_v0(SparseTensor& input1, SparseTensor& input2, const templateFn<T>& f) {
    TORCH_CHECK(input1.range() == input2.range());
    TORCH_CHECK(input1.dtype() == input2.dtype());
    auto shape1 = input1.indices().sizes().vec();
    auto shape2 = input2.indices().sizes().vec();
    auto sparse_dim1 = input1.sparse_dim();
    auto sparse_dim2 = input2.sparse_dim();
    TORCH_CHECK(shape1.size() - sparse_dim1 == shape2.size() - sparse_dim2); // align from tailling
    int64_t sparse_dim_from_tailing = shape1.size() - sparse_dim1 - 1;
    auto in_id_ptr1 = indices_ptr(input1);
    auto in_val_ptr1 = values_ptr<T>(input1);
    auto in_id_ptr2 = indices_ptr(input2);
    auto in_val_ptr2 = values_ptr<T>(input2);
    int sparse_size1 = input1._size(sparse_dim1);
    int sparse_size2 = input2._size(sparse_dim2);
    // find out related shapes
    int out_dims = max(shape1.size(), shape2.size());
    int out_sparse_dim = out_dims - sparse_dim_from_tailing - 1;
    auto unrelated_shape = vector<int64_t>(out_dims, 0); // unrelated_shape[sparse_dim] = 1
    bool input1_broadcast = false, input2_broadcast = false;
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
        auto max_size = max(size1, size2);
        unrelated_shape[out_dims - tl - 1] = max_size;
        max_size > size1 && (input1_broadcast = true);
        max_size > size2 && (input2_broadcast = true);
    }
    // find max_sparse_size
    int64_t out_sparse_size = 0;
    sizes_for(unrelated_shape, true, [&](auto ids, auto global) {
        // already ids[out_sparse_dim] = 0
        unordered_set<int64_t> id_set;
        for (const auto i : c10::irange(sparse_size1)) {
            ids[out_sparse_dim] = i;
            auto id = local2global_offset_broadcast(shape1, ids, out_dims);
            id_set.insert(in_id_ptr1[id]);
        }
        for (const auto i : c10::irange(sparse_size2)) {
            ids[out_sparse_dim] = i;
            auto id = local2global_offset_broadcast(shape2, ids, out_dims);
            id_set.insert(in_id_ptr2[id]);
        }
        out_sparse_size = max(out_sparse_size, (int64_t)id_set.size());
    });
    // skip when one is 0, another one is broadcast.
    const templateFn<T>& broadcast_f = [&f, &input1_broadcast, &input2_broadcast](auto a, auto b) {
        if ((a == 0 && input2_broadcast) || (b == 0 && input1_broadcast)) return static_cast<T>(0);
        return f(a, b);
    };
    // init out tensor
    vector<int64_t> out_shape = unrelated_shape;
    out_shape[out_sparse_dim] = out_sparse_size;
    auto out = create_sparse_IdVal_options<T>(out_shape, input1.values().options());
    // elementwise operation
    sizes_for(unrelated_shape, true, [&](auto ids, auto global) {
        // already ids[out_sparse_dim] = 0
        unordered_map<int64_t, T> out_map;
        unordered_set<int64_t> out_set;
        for (const auto i : c10::irange(sparse_size1)) {
            ids[out_sparse_dim] = i;
            auto offset = local2global_offset_broadcast(shape1, ids, out_dims);
            auto id = in_id_ptr1[offset];
            out_map[id] = in_val_ptr1[offset];
            out_set.insert(id);
        }
        for (const auto i : c10::irange(sparse_size2)) {
            ids[out_sparse_dim] = i;
            auto offset = local2global_offset_broadcast(shape2, ids, out_dims);
            auto id = in_id_ptr2[offset];
            auto num1 = out_map[id];
            auto num2 = in_val_ptr2[offset];
            out_map[id] = broadcast_f(num1, num2);
            out_set.erase(id);
        }
        // when num1 != 0, num2 == 0
        for (const auto& id : out_set) {
            out_map[id] = broadcast_f(out_map[id], 0);
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

template SparseTensor elementwise_template_v0<float>(
    SparseTensor& input1, SparseTensor& input2, const templateFn<float>& f);
template SparseTensor elementwise_template_v0<double>(
    SparseTensor& input1, SparseTensor& input2, const templateFn<double>& f);


/**
 * currently, union of two inputs at sparse_dim.
 * Broadcast rules follow: https://pytorch.org/docs/stable/notes/broadcasting.html
 * @param f (float/double, float/double) -> float/double, operation of each pair.
 */
template<typename T>
SparseTensor elementwise_template(SparseTensor& input1, SparseTensor& input2, const templateFn<T>& f) {
    TORCH_CHECK(input1.is_coalesced() && input2.is_coalesced(), "input1 and input2 should coalesce() before");
    TORCH_CHECK(input1.dtype() == input2.dtype(), "inputs should be same dtype");
    TORCH_CHECK(input1.range() == input2.range(), "inputs should be same id range");
    auto shape1 = input1.indices().sizes().vec();
    auto shape2 = input2.indices().sizes().vec();
    TORCH_CHECK(shape1.size() - input1.sparse_dim() == shape2.size() - input2.sparse_dim(), 
        "input1 and input2 must be same dims from sparse_dim to end");

    // calculate unrelated_shape (), and unrelated_shape[sparse_dim] = 1
    int64_t sparse_dim_from_tailing = shape1.size() - input1.sparse_dim() - 1;
    int out_dims = max(shape1.size(), shape2.size());
    int out_sparse_dim = out_dims - sparse_dim_from_tailing - 1;
    auto unrelated_shape = vector<int64_t>(out_dims, 0);
    bool input1_broadcast = false, input2_broadcast = false;
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
        auto max_size = max(size1, size2);
        unrelated_shape[out_dims - tl - 1] = max_size;
        max_size > size1 && (input1_broadcast = true);
        max_size > size2 && (input2_broadcast = true);
    }

    // create out tensor (assume the worst case: any id from two inputs are same)
    vector<int64_t> out_shape = unrelated_shape;
    out_shape[out_sparse_dim] = input1._size(input1.sparse_dim()) + input2._size(input2.sparse_dim());
    auto out = create_sparse_IdVal_options<T>(out_shape, input1.values().options());

    // skip when one is 0, another one is broadcast.
    const templateFn<T>& broadcast_f = [&f, &input1_broadcast, &input2_broadcast](auto a, auto b) {
        if ((a == 0 && input2_broadcast) || (b == 0 && input1_broadcast)) return static_cast<T>(0);
        return f(a, b);
    };

    // prepare read/write pointers
    auto in_id_ptr1 = indices_ptr(input1), in_val_ptr1 = values_ptr<T>(input1);
    auto in_id_ptr2 = indices_ptr(input2), in_val_ptr2 = values_ptr<T>(input2);
    // count each sparse_size, => maximum sparse_size
    auto sparse_size_count = torch::zeros(unrelated_shape, torch::dtype(torch::kI64));
    auto sparse_size_count_ptr = sparse_size_count.data_ptr<int64_t>();
    // stride for sparse_dim (prod of dims from [sparse_dim (not include), end])
    auto stride1 = accumulate(shape1.begin() + input1.sparse_dim() + 1, shape1.end(), 1, multiplies<int64_t>());
    auto stride2 = accumulate(shape2.begin() + input2.sparse_dim() + 1, shape2.end(), 1, multiplies<int64_t>());
    auto stride_out = accumulate(out_shape.begin() + out_sparse_dim + 1, out_shape.end(), 1, multiplies<int64_t>());

    auto sparse_size1 = input1._size(input1.sparse_dim());
    auto sparse_size2 = input2._size(input2.sparse_dim());
    sizes_for(unrelated_shape, true, [&](auto ids, auto global) {
        // start global offset
        auto start1 = local2global_offset_broadcast(shape1, ids, out_dims);
        auto start2 = local2global_offset_broadcast(shape2, ids, out_dims);
        auto end1 = start1 + sparse_size1 * stride1;
        auto end2 = start2 + sparse_size2 * stride2;
        auto p1 = start1, p2 = start2;
        auto p_start = local2global_offset(out_shape, ids);
        auto p_out = p_start;
        // tranverasal (merge) sorted two array
        while (p1 < end1 && p2 < end2) {
            if (in_id_ptr1[p1] < in_id_ptr2[p2]) {
                out.id_ptr[p_out] = in_id_ptr1[p1];
                out.val_ptr[p_out] = broadcast_f(in_val_ptr1[p1], 0);
                p1 += stride1;
            }
            else if (in_id_ptr1[p1] > in_id_ptr2[p2]) {
                out.id_ptr[p_out] = in_id_ptr2[p2];
                out.val_ptr[p_out] = broadcast_f(0, in_val_ptr2[p2]);
                p2 += stride2;
            }
            else {
                out.id_ptr[p_out] = in_id_ptr1[p1]; // no matter p1 or p2
                out.val_ptr[p_out] = broadcast_f(in_val_ptr1[p1], in_val_ptr2[p2]);
                p1 += stride1;
                p2 += stride2;
            }

            out.val_ptr[p_out] != 0 && (p_out += stride_out);
        }
        while (p1 < end1) {
            out.id_ptr[p_out] = in_id_ptr1[p1];
            out.val_ptr[p_out] = broadcast_f(in_val_ptr1[p1], 0);
            p1 += stride1;
            out.val_ptr[p_out] != 0 && (p_out += stride_out);
        }
        while (p2 < end2) {
            out.id_ptr[p_out] = in_id_ptr2[p2];
            out.val_ptr[p_out] = broadcast_f(0, in_val_ptr2[p2]);
            p2 += stride2;
            out.val_ptr[p_out] != 0 && (p_out += stride_out);
        }

        // sparse_size
        sparse_size_count_ptr[global] = (p_out - p_start) / stride_out;
    });

    // slice out tensor by maximum sparse_size (remove all same-len zeros tailing)
    auto max_len = sparse_size_count.max().item<int64_t>();
    out.indices = out.indices.narrow(out_sparse_dim, 0, max_len).contiguous();
    out.values = out.values.narrow(out_sparse_dim, 0, max_len).contiguous();

    auto out_tensor = SparseTensor(out.indices, out.values, out_sparse_dim, input1.range());
    out_tensor.c_set_coalesced(true); // guarantee that sorted merge array, also is sorted, unique array.
    return out_tensor;
}

template SparseTensor elementwise_template<float>(
    SparseTensor& input1, SparseTensor& input2, const templateFn<float>& f);
template SparseTensor elementwise_template<double>(
    SparseTensor& input1, SparseTensor& input2, const templateFn<double>& f);


/**
 * @param input2 _indices: [sparse_dims, nse], _values: [nse, dense_dims]
 */
template<typename T>
SparseTensor elementwise_template(
    SparseTensor& input1, 
    torch::Tensor& input2, 
    const templateFn<T>& f,
    bool inplace
) {
    TORCH_CHECK(input2.layout() == torch::kSparse); // sparse_coo_tensor
    TORCH_CHECK(input1.dtype() == input2.dtype());

    auto indices1 = input1.indices();
    auto values1 = input1.values();
    auto id1_sizes = indices1.sizes().vec();
    auto val1_sizes = values1.sizes().vec();
    auto id1_ptr = indices_ptr(input1);
    auto val1_ptr = values_ptr<T>(input1);
    auto indices2 = input2._indices().contiguous();
    auto id2_ptr = indices2.data_ptr<int64_t>();
    auto values2 = input2._values().contiguous();
    auto val2_ptr = values2.data_ptr<T>();
    auto sparse_dim = input1.sparse_dim();
    auto sparse_dims2 = indices2.size(0);
    auto dense_dims2 = input2.dense_dim();
    // calculate dense size of input1 and input2
    int64_t dense1_size = 1;
    for (const auto i : c10::irange(indices1.dim(), values1.dim())) 
        dense1_size *= input1._size(i);
    int64_t dense2_size = 1;
    for (const auto i : c10::irange(dense_dims2))
        dense2_size *= input2._values().size(1 + i); // size(0) is nse
    
    // input2 sparse_dims should cover input1 sparse_dims
    // additional dims towards front part of input1.dense_dims
    TORCH_CHECK(sparse_dims2 + dense_dims2 == values1.dim());
    auto sparse_dims1 = indices1.dim();
    TORCH_CHECK(input1.indices().dim() <= sparse_dims2);
    // TORCH_CHECK(dense1_size == dense2_size, "input1 and input2 dense part should be exactly same.");

    auto nse = indices2.size(1);
    auto id1_locals = vector<int64_t>(sparse_dims1, 0);
    auto val1_locals = vector<int64_t>(values1.dim(), 0);

    auto out = input1;
    if (!inplace) {
        auto ids = indices1.clone().detach();
        auto vals = values1.clone().detach();
        out = SparseTensor(ids, vals, sparse_dim, input1.range());
    }
    // auto out_id_ptr = indices_ptr(out);
    auto out_val_ptr = out.values().contiguous().data_ptr<T>();
    for (const auto i : c10::irange(nse)) {
        // traversal along the sparse dim, find sparse1_id == sparse2_id
        for (const auto j : c10::irange(input1._size(sparse_dim))) {
            // fill beginning part of locals of input2
            for (const auto k : c10::irange(sparse_dims1)) {
                auto id = id2_ptr[k * nse + i];
                id1_locals[k] = id;
                val1_locals[k] = id;
                TORCH_CHECK(id >= 0, "indices of input2 should >= 0.");
                TORCH_CHECK(k == sparse_dim ? id < input1.range() : id < input1._size(k),
                    "indices of input2 exceed input1 range.");
            }
            // fill rest of locals of input2
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
                for (const auto z : c10::irange(dense2_size)) {
                    out_val_ptr[offset + z] = f(val1_ptr[offset + z], val2_ptr[i * dense2_size + z]);
                }
            }
        }
    }

    return out;
}

template SparseTensor elementwise_template<float>(
    SparseTensor& input1, torch::Tensor& input2, const templateFn<float>& f, bool inplace);
template SparseTensor elementwise_template<double>(
    SparseTensor& input1, torch::Tensor& input2, const templateFn<double>& f, bool inplace);


/**
 * reshape only merge sparse_dim, not divide into multiple dims.
 * no sorted, only transform id.
 */
SparseTensor reshape(SparseTensor& input, vector<int64_t>& out_shape) {
    auto in_sizes = input.sizes();
    auto in_sparse_dim = input.sparse_dim();
    auto in_sparse_size = in_sizes[in_sparse_dim]; // actually is input.range()
    // in_dsd: [prev dense_dim, sparse_dim, next dense_dim]
    vector<int64_t> in_dsd = {
        accumulate(in_sizes.begin(), in_sizes.begin() + in_sparse_dim, 1, multiplies<int64_t>()),
        in_sparse_size,
        accumulate(in_sizes.begin() + in_sparse_dim + 1, in_sizes.end(), 1, multiplies<int64_t>())
    };

    // in_sparse_dim -> out_sparse_dim
    auto prev_acc = 1;
    auto out_sparse_dim = -1;
    for (int i = 0; i < (int)out_shape.size(); i++) {
        prev_acc *= out_shape[i];
        if (prev_acc > in_dsd[0]) {
            out_sparse_dim = i;
            break;
        }
    }
    TORCH_CHECK(out_sparse_dim >= 0);
    auto out_sparse_size = out_shape[out_sparse_dim];
    TORCH_CHECK(out_sparse_size % in_sparse_size == 0, "output sparse size must times of input sparse size");
    // input can be divided by: [-1, out_sparse_size=[sparse_prev_part, in_sparse_size, -1], last_dense_part]
    auto sparse_prev_part = in_dsd[0] / (prev_acc / out_sparse_size);
    auto sparse_next_part = out_sparse_size / in_sparse_size / sparse_prev_part;
    auto last_dense_part = in_dsd[2] / sparse_next_part;
    auto sparse_dsd = vector<int64_t>{sparse_prev_part, in_sparse_size, sparse_next_part};

    // be careful to distinguish indices size and size (range)
    auto out_range = input.range() * sparse_prev_part * sparse_next_part;
    auto in_id_sizes = input.indices().sizes().vec();
    vector<int64_t> out_id_sizes = out_shape;
    out_id_sizes[out_sparse_dim] = in_id_sizes[in_sparse_dim] * sparse_prev_part * sparse_next_part;
    auto out_ids = torch::zeros(out_id_sizes, input.indices().options());
    TORCH_CHECK(out_ids.numel() == input.indices().numel(), "out shape of reshape is not same with the input.");
    auto out_id_ptr = out_ids.data_ptr<int64_t>();
    TORCH_CHECK(input.indices().numel() == input.values().numel());
    auto out_vals = input.values().view(out_id_sizes);
    auto in_id_ptr = indices_ptr(input);
    in_dsd[1] = in_id_sizes[in_sparse_dim];
    sizes_for(in_dsd, true, [&](auto locals, auto global) {
        auto out_sparse_ids = vector<int64_t>(3, 0);
        out_sparse_ids[0] = locals[0] % sparse_prev_part;
        out_sparse_ids[1] = in_id_ptr[global];
        out_sparse_ids[2] = locals[2] / last_dense_part;
        out_id_ptr[global] = local2global_offset(sparse_dsd, out_sparse_ids);
    });

    return SparseTensor(out_ids, out_vals, out_sparse_dim, out_range);
}

